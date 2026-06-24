import asyncio
import redis.asyncio as aioredis

REDIS_URL = "redis://localhost:6379/0"

async def seed():
    print("Connecting to Redis to seed initial provider metrics...")
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    
    mock_data = {
        "provider:groq": {
            "ttft": "180.0",
            "error_rate": "0.01",
            "circuit_status": "closed",
            "failures_count": "0"
        },
        "provider:together": {
            "ttft": "240.0",
            "error_rate": "0.02",
            "circuit_status": "closed",
            "failures_count": "0"
        },
        "provider:deepseek": {
            "ttft": "450.0",
            "error_rate": "0.005",
            "circuit_status": "closed",
            "failures_count": "0"
        }
    }
    
    async with r.pipeline(transaction=False) as pipe:
        for key, fields in mock_data.items():
            pipe.hset(key, mapping=fields)
        await pipe.execute()
        
    print("Successfully seeded provider states! Closing connection.")
    await r.aclose()

if __name__ == "__main__":
    asyncio.run(seed())