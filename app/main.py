import sys

from fastapi import FastAPI
import uvicorn
import logging
from app.config import Config
import app.dependencies as dependencies
from app.external_downloader import download_nltk_model
from app.routers.search_router import SearchRouter
from app.services.search_engine import SearchEngine

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(name=Config.LOGGER_NAME)

logger.info("Downloading NLTK models from the registry...")
download_nltk_model(logger=logger, models=["stopwords", "wordnet"])
# Initialize FastAPI
app = FastAPI(
    title=Config.APP_NAME,
    description="A FastAPI-powered search engine to search 20-newsgroups dataset",
    version="1.0.0"
)

logger.info(f"{Config.APP_NAME}: Service Initialised")

if Config.USE_REDIS:
    redis_client = dependencies.connect_redis(
        host=Config.REDIS_HOST,
        port=Config.REDIS_PORT,
        username=Config.REDIS_USERNAME,
        password=Config.REDIS_PASSWORD,
        ssl=Config.REDIS_SSL
    )
    redis_client.set("key", "value", ex=Config.REDIS_TTL)
else:
    redis_client = None
# Load search engine
search_engine = SearchEngine(logger=logger)
# Initialize search router
search_router = SearchRouter(redis_client=redis_client, search_engine=search_engine, logger=logger)
app.include_router(search_router.router)

# Load the model via initialize class

if __name__ == "__main__":
    uvicorn.run(app="app.main:app", host="0.0.0.0", port=8000, log_level="info")
