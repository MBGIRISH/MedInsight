from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import ingest, audit, health
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('medinsight.log')
    ]
)

app = FastAPI(
    title="MedInsight API",
    description="AI Clinical Notes Auditor - Medical Audit System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MedInsight API - AI Clinical Notes Auditor",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "ingest": "/api/ingest",
            "audit": "/api/audit",
            "get_audit": "/api/audit/{audit_id}"
        }
    }

# Include routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(ingest.router, prefix="/api", tags=["Ingestion"])
app.include_router(audit.router, prefix="/api", tags=["Audit"])

# Analytics router
from app.routes import analytics
app.include_router(analytics.router, prefix="/api", tags=["Analytics"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("MedInsight API starting up...")
    # Initialize MongoDB connection
    from app.models.db import MongoDB
    MongoDB.connect()
    print("MongoDB connected")
    print("Vector store will be initialized on first use")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    from app.models.db import MongoDB
    MongoDB.close()
    print("MedInsight API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

