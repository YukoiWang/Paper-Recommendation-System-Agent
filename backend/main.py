"""
IntelliScholar Backend Main Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.api import router
from backend.config import settings

app = FastAPI(
    title="IntelliScholar API",
    description="多智能体论文推荐系统",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("IntelliScholar API starting up...")
    # TODO: Initialize database connections, load models, etc.


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("IntelliScholar API shutting down...")


@app.get("/")
async def root():
    return {"message": "IntelliScholar API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
