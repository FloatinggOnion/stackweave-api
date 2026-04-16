"""FastAPI application for Stackweave solver."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routes import router

# Create FastAPI app
app = FastAPI(
    title="Stackweave Solver API",
    description="FastAPI backend for dependency conflict resolution",
    version="0.1.0",
)

# Configure CORS for development
# Allow localhost:3000 (Next.js dev) and any origin for now
# Will tighten in Phase 2
app.add_middleware(
    CORSMiddleware,
    allow_origins=["localhost:3000", "http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include solve router
app.include_router(router)


@app.get("/")
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "ok"})


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
