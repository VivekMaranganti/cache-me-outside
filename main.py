import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

load_dotenv()

REDIS_URL = "redis://localhost:6379/0"
HTTP_TIMEOUT_SECONDS = 10.0
CIRCUIT_FAILURE_THRESHOLD = 5
CIRCUIT_HALF_OPEN_AFTER_S = 30
EMA_ALPHA = 0.2

PROVIDERS_CONFIG = {
    #Information that the router needs to know about the provider
    "groq": {
        #Endpoint that the client will post to when the routing decision is made
        "url": "https://api.groq.com/openai/v1/chat/completions",

        #Reads from .env file to grab the api key
        "api_key": os.getenv("GROQ_API_KEY", ""),

        #Model accuracy and token usage ratio based on splitting by the model
        "accuracy": 0.82,
        "token_inflation": 1.0,

        #Model name
        "default_model": "llama-3.1-8b-instant",
    },
    "together": {
        "url": "https://api.together.xyz/v1/chat/completions",
        "api_key": os.getenv("TOGETHER_API_KEY", ""),
        "accuracy": 0.85,
        "token_inflation": 1.1,
        "default_model": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
    },
    "deepseek": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "accuracy": 0.89,
        "token_inflation": 0.95,
        "default_model": "deepseek-chat",
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Initializing persistent connection pools...")
    #Creates a redis connection pool
    app.state.redis = aioredis.from_url(REDIS_URL, decode_responses=True, max_connections=50)
    #50 persistent connections to providers
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
    #Creates the HTTP client
    app.state.http_client = httpx.AsyncClient(limits=limits, timeout=HTTP_TIMEOUT_SECONDS)
    yield
    print("[SHUTDOWN] Closing persistent connection pools...")
    #Sharing objects across requests
    await app.state.redis.aclose()
    await app.state.http_client.aclose()


app = FastAPI(lifespan=lifespan)


async def telemetry_success(redis_client, provider: str, ttft_ms: float, total_ms: float):
    try:
        #Builds redis hash key
        key = f"provider:{provider}"
        stats = await redis_client.hgetall(key)
        #reads EMA values
        old_ttft = float(stats.get("ttft", 300.0))
        old_error_rate = float(stats.get("error_rate", 0.0))
        #Calculates new EMA
        new_ttft = EMA_ALPHA * ttft_ms + (1 - EMA_ALPHA) * old_ttft
        new_error_rate = (1 - EMA_ALPHA) * old_error_rate
        #Writes all updated fields to redis
        await redis_client.hset(key, mapping={
            "ttft": str(round(new_ttft, 3)),
            "error_rate": str(round(new_error_rate, 6)),
            "circuit_status": "closed",
            "failures_count": "0",
            "circuit_opened_at": "0",
        })
        print(f"[TELEMETRY] {provider}: TTFT={ttft_ms:.1f}ms→EMA {new_ttft:.1f}ms | err_rate→{new_error_rate:.4f}")
    except Exception as e:
        print(f"[TELEMETRY ERROR] {provider} success write-back failed: {e}")


async def telemetry_failure(redis_client, provider: str):
    try:
        key = f"provider:{provider}"
        #Builds redis hash key
        stats = await redis_client.hgetall(key)
        #Reads EMA values
        old_error_rate = float(stats.get("error_rate", 0.0))
        failures = int(stats.get("failures_count", 0)) + 1
        #Calculates new EMA values
        new_error_rate = EMA_ALPHA * 1.0 + (1 - EMA_ALPHA) * old_error_rate
        mapping = {
            "error_rate": str(round(new_error_rate, 6)),
            "failures_count": str(failures),
        }
        #Guards against consecutive failures
        if failures >= CIRCUIT_FAILURE_THRESHOLD:
            mapping["circuit_status"] = "open"
            mapping["circuit_opened_at"] = str(time.time())
            print(f"[CIRCUIT BREAKER] {provider.upper()} TRIPPED after {failures} consecutive failures")
        else:
            print(f"[TELEMETRY] {provider}: failure {failures}/{CIRCUIT_FAILURE_THRESHOLD} | err_rate→{new_error_rate:.4f}")
        await redis_client.hset(key, mapping=mapping)
    except Exception as e:
        print(f"[TELEMETRY ERROR] {provider} failure write-back failed: {e}")


def is_routable(provider: str, metrics: dict) -> bool:
    #Read current providers redis metrics
    status = metrics.get("circuit_status", "closed")
    if status == "closed":
        return True
    #Check if cooldown period has ended if circuit failed
    if status == "open":
        opened_at = float(metrics.get("circuit_opened_at", 0))
        if time.time() - opened_at >= CIRCUIT_HALF_OPEN_AFTER_S:
            print(f"[CIRCUIT BREAKER] {provider} entering HALF-OPEN — sending recovery probe")
            return True
        return False
    if status == "half_open":
        return True
    return True


async def stream_from_provider(provider, cfg, body, http_client, redis_client):
    #use model from request body, fall back to provider default if not specified
    model = body.get("model") or cfg["default_model"]
    #spread original request body, override model and force streaming on
    payload = {**body, "model": model, "stream": True}
    #auth header
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    start = time.perf_counter()
    ttft_ms = None
    try:
        #open streaming connection to provider
        async with http_client.stream("POST", cfg["url"], json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                #skip blank lines, SSE uses them as chunk separators
                if not raw_line:
                    continue
                #first data line means first token arrived, record TTFT
                if ttft_ms is None and raw_line.startswith("data:"):
                    ttft_ms = (time.perf_counter() - start) * 1000
                #relay chunk directly to client
                yield (raw_line + "\n\n").encode()
                if raw_line.strip() == "data: [DONE]":
                    break
        total_ms = (time.perf_counter() - start) * 1000
        #fire telemetry as background task, client doesnt wait for this
        asyncio.create_task(telemetry_success(redis_client, provider, ttft_ms or 0.0, total_ms))
    except Exception as e:
        print(f"[PROVIDER ERROR] {provider} failed: {type(e).__name__}: {e}")
        #increment failure count, trips circuit at threshold
        asyncio.create_task(telemetry_failure(redis_client, provider))
        err_chunk = json.dumps({"error": {"message": f"Provider {provider} failed", "type": "upstream_error"}})
        yield f"data: {err_chunk}\n\n".encode()
        yield b"data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def route_llm_request(request: Request):
    redis_client = request.app.state.redis
    http_client = request.app.state.http_client

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    start_routing = time.perf_counter()
    provider_keys = list(PROVIDERS_CONFIG.keys())

    #pull all provider stats in a single redis round trip
    async with redis_client.pipeline(transaction=False) as pipe:
        for provider in provider_keys:
            pipe.hgetall(f"provider:{provider}")
        raw_stats = await pipe.execute()

    live_metrics = {provider_keys[i]: raw_stats[i] for i in range(len(provider_keys))}
    scored = []

    #score each provider, skip any with open circuits
    for provider, cfg in PROVIDERS_CONFIG.items():
        metrics = live_metrics.get(provider, {})
        if not is_routable(provider, metrics):
            print(f"[ROUTER] Skipping {provider} — circuit OPEN")
            continue
        live_ttft = float(metrics.get("ttft", 300.0))
        live_error_rate = float(metrics.get("error_rate", 0.0))
        #lower score wins
        score = ((live_ttft * cfg["token_inflation"]) / cfg["accuracy"]) + (live_error_rate * 5000)
        print(f"[EVAL] {provider}: TTFT={live_ttft}ms, err={live_error_rate:.4f}, score={score:.2f}")
        scored.append((score, provider))

    #sort ascending, lowest score is best provider
    scored.sort(key=lambda x: x[0])

    if not scored:
        raise HTTPException(status_code=503, detail="All upstream providers are unavailable")

    routing_overhead_ms = (time.perf_counter() - start_routing) * 1000
    best_provider = scored[0][1]
    #remaining providers in score order, used as fallback reference
    fallback_order = [p for _, p in scored]

    print(f"[ROUTER] Selected: {best_provider.upper()} (overhead: {routing_overhead_ms:.2f}ms | fallback order: {fallback_order})")

    return StreamingResponse(
        stream_from_provider(best_provider, PROVIDERS_CONFIG[best_provider], body, http_client, redis_client),
        media_type="text/event-stream",
        #expose routing decision and overhead in response headers
        headers={
            "X-Routed-To": best_provider,
            "X-Routing-Overhead-Ms": f"{routing_overhead_ms:.2f}",
            "Cache-Control": "no-cache",
        },
    )


@app.get("/health")
async def health(request: Request):
    redis_client = request.app.state.redis
    provider_keys = list(PROVIDERS_CONFIG.keys())
    #single pipelined round trip for all provider states
    async with redis_client.pipeline(transaction=False) as pipe:
        for p in provider_keys:
            pipe.hgetall(f"provider:{p}")
        raw = await pipe.execute()
    stats = {provider_keys[i]: raw[i] for i in range(len(provider_keys))}
    return {"status": "ok", "providers": stats}