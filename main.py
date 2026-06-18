import asyncio
from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import redis.asyncio as aioredis
import httpx

REDIS_URL = "redis://localhost:6379/0"
HTTP_TIMEOUT_SECONDS = 5.0

PROVIDERS_CONFIG = {
    "groq": {"url": "https://api.groq.com/openai/v1/chat/completions", "accuracy": 0.82, "token_inflation": 1.0},
    "together": {"url": "https://api.together.xyz/v1/chat/completions", "accuracy": 0.85, "token_inflation": 1.1},
    "deepseek": {"url": "https://api.deepseek.com/v1/chat/completions", "accuracy": 0.89, "token_inflation": 0.95}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Initializing persistent connection pools...")
    app.state.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
    app.state.http_client = httpx.AsyncClient(limits=limits, timeout=HTTP_TIMEOUT_SECONDS)
    yield
    print("[SHUTDOWN] Closing persistent connection pools...")
    await app.state.redis.close()
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)

async def update_provider_telemetry(provider: str, ttft: float, status_code: int):
    try:
        print(f"[TELEMETRY LOG] Async update for {provider}: TTFT={ttft:.2f}ms, Status={status_code}")
        pass
    except Exception as e:
        print(f"[TELEMETRY ERROR] Failed to write back to Redis: {e}")

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
    
    async with redis_client.pipeline(transaction=False) as pipe:
        for provider in provider_keys:
            pipe.hgetall(f"provider:{provider}")
        raw_stats = await pipe.execute()
    
    live_metrics = {provider_keys[i]: raw_stats[i] for i in range(len(provider_keys))}
    
    best_provider = None
    best_score = float("inf")

    for provider, static_cfg in PROVIDERS_CONFIG.items():
        metrics = live_metrics.get(provider, {})
        
        if metrics.get("circuit_status") == "open":
            print(f"[ROUTER] Skipping {provider} - Circuit Breaker is OPEN.")
            continue
            
        live_ttft = float(metrics.get("ttft", 300.0))
        live_error_rate = float(metrics.get("error_rate", 0.0))
        
        accuracy_factor = static_cfg["accuracy"]
        token_multiplier = static_cfg["token_inflation"]
        error_penalty = live_error_rate * 5000
        
        score = ((live_ttft * token_multiplier) / accuracy_factor) + error_penalty
        
        print(f"[EVAL] {provider} -> Live TTFT: {live_ttft}ms, Score: {score:.2f}")
        
        if score < best_score:
            best_score = score
            best_provider = provider

    if not best_provider:
        best_provider = "groq" 
        
    chosen_url = PROVIDERS_CONFIG[best_provider]["url"]
    routing_overhead = (time.perf_counter() - start_routing) * 1000
    print(f"[ROUTER] Target selected: **{best_provider.upper()}** (Decision optimized in {routing_overhead:.2f}ms)")

    async def mock_stream_generator():
        yield f"data: {{'choices': [{{'delta': {{'content': 'Routed live to {best_provider}!'}}}}]}}\n\n".encode()
        await asyncio.sleep(0.05)
        asyncio.create_task(update_provider_telemetry(best_provider, ttft=145.0, status_code=200))
        yield b"data: [DONE]\n\n"

    return StreamingResponse(mock_stream_generator(), media_type="text/event-stream")