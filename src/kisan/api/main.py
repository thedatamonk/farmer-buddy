"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kisan.api.routes import chat, health, search
from kisan.core.config import get_settings
from kisan.core.logging import logger, setup_logging
from kisan.modules.mandi.repository import MandiPriceRepository
from kisan.modules.mandi.scheduler import MandiPriceFetchScheduler
from kisan.services.database import DatabaseService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    setup_logging()
    logger.info("Starting Kisan API server")

    settings = get_settings()

    # Initialize database service
    db_service = DatabaseService(settings)
    try:
        await db_service.connect()
        app.state.db_service = db_service
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")
        app.state.db_service = db_service  # Keep reference even if not connected

    # Initialize mandi repository and scheduler
    app.state.mandi_repository = None
    app.state.mandi_scheduler = None

    if db_service.is_connected():
        # Create repository
        mandi_repository = MandiPriceRepository(db_service, settings)
        app.state.mandi_repository = mandi_repository

        # Start scheduler
        mandi_scheduler = MandiPriceFetchScheduler(mandi_repository, settings)
        await mandi_scheduler.start()
        app.state.mandi_scheduler = mandi_scheduler
        logger.info("Mandi price scheduler started")
    else:
        logger.warning("Database not connected, mandi caching disabled")

    yield

    # Shutdown
    logger.info("Shutting down Kisan API server")

    # Stop scheduler
    if app.state.mandi_scheduler:
        await app.state.mandi_scheduler.stop()

    # Disconnect database
    if app.state.db_service:
        await app.state.db_service.disconnect()


app = FastAPI(
    title="Kisan API",
    description="Agricultural chatbot API for Indian farmers",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])
