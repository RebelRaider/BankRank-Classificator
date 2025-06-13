from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .core.config import settings
from .core.database import engine, Base
from .api.endpoints import classification, analytics
from .utils.logging import setup_logging
from .api.endpoints import files

# Create database tables
Base.metadata.create_all(bind=engine)

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Автоматическая классификация текстовых обращений банков с анализом токсичности и рейтингов"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],  # React dev server and Docker
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    classification.router,
    prefix=f"{settings.API_V1_STR}/classification",
    tags=["classification"]
)

app.include_router(
    analytics.router,
    prefix=f"{settings.API_V1_STR}/analytics",
    tags=["analytics"]
)

app.include_router(
    files.router,
    prefix=f"{settings.API_V1_STR}/files",
    tags=["files"]
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Bank Classification Service API",
        "version": settings.VERSION,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {"status": "healthy", "service": settings.PROJECT_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
