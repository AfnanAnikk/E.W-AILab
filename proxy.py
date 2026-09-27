import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("energywise-proxy")

app = FastAPI(title="GridWise Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GRIDWISE_URL = "https://gridwise-api.onrender.com"

# Generous timeout: 90s read (covers Render cold start + Gemini LLM inference), 30s connect
HTTPX_TIMEOUT = httpx.Timeout(90.0, connect=30.0)


@app.get("/health")
async def health():
    return {"status": "ok", "proxy": "running"}


@app.api_route("/gridwise/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def gridwise_proxy(path: str, request: Request):
    if request.method == "OPTIONS":
        return Response(status_code=204)

    target_url = f"{GRIDWISE_URL}/{path}"
    body = await request.body()
    headers = {"Content-Type": request.headers.get("content-type", "application/json")}

    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            if request.method == "POST":
                response = await client.post(target_url, content=body, headers=headers)
            else:
                response = await client.get(target_url, headers=headers)

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers={
                "Content-Type": response.headers.get("content-type", "application/json"),
                "Access-Control-Allow-Origin": "*",
            },
        )
    except httpx.TimeoutException:
        logger.warning(f"Timeout connecting to GridWise at {target_url}")
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={"detail": "GridWise API timed out. Render may be waking up from sleep or LLM is busy. Please retry in a few seconds."}
        )
    except Exception as e:
        logger.error(f"Proxy error connecting to GridWise: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": f"Proxy communication error: {str(e)}"}
        )