import json
from logging import Logger
from typing import List, Dict

import redis as redis
import hashlib
from fastapi import APIRouter, Depends
from app.config import Config
from fastapi.responses import ORJSONResponse
from app.dependencies import verify_key
from app.models.IO_model import SearchInputModel, AddInputModel, SearchOutputModel
from app.services.search_engine import SearchEngine
import time


class SearchRouter:
    def __init__(self,  logger: Logger, search_engine: SearchEngine, redis_client=None) -> None:
        self.__redis_client: redis.client.Redis = redis_client
        self.__logger = logger
        self.__searcher = search_engine

    def __set_key(self, key: str, value: List[str]):
        try:
            value = json.dumps(value)
            self.__redis_client.set(key, value, ex=Config.REDIS_TTL)
        except Exception as e:
            self.__logger.error(f"Error saving values to redis. Error: {repr(e)}")
            pass

    def __get_key(self, key: str):
        try:
            cached_result = self.__redis_client.get(key)
        except Exception as e:
            self.__logger.error(f"Error getting keys from redis. Error: {repr(e)}")
            cached_result = None
            pass
        return cached_result

    # Hash the term to use as a Redis key
    def __hash_term(self, term: str) -> str:
        return hashlib.sha256(term.encode('utf-8')).hexdigest()

    @property
    def router(self):
        api_router = APIRouter(
            prefix=Config.BASE_API_PATH, tags=["ranking-search-engine"]
        )

        @api_router.get("/status")
        def status():
            return {"status": 200, "ok": True}

        @api_router.post(
            path="/search",
            dependencies=[Depends(verify_key)],
            response_class=ORJSONResponse,
            description="searches for top 10 most similar results in the DB based on TF-IDF scoring method"
        )
        def search(search_term: SearchInputModel):
            process_start = time.time()
            term = search_term.search_entry
            if Config.USE_REDIS:
                self.__logger.info(f"Redis feature not fully implemented")
                """
                Workflow - 
                1. hash the term and see if it is already stored in Redis for fast fetching of results
                2. If it is not there do a common search and do a redis set, hashing the entry to reduce dimensions and 
                   using TTL param which will delete the newer used search terms in a reasonable time to reduce redis 
                   size and cost, store hashed entry as key and results as a value
                3. In time this would reduce the computation time significantly 
                """

            document_indexes = self.__searcher.search_tfidf(query=term)
            results: Dict[str: List[str]] = {"document_indexes": document_indexes}
            try:
                SearchOutputModel.model_validate(results)
            except Exception as e:
                self.__logger.error(f"Output validation failed. Error: {repr(e)}")
            process_end = time.time()
            self.__logger.info(f"Time to search entry: {round((process_end-process_start), 3)} ms")
            return ORJSONResponse(results)

        @api_router.post(
            path="/add_entry",
            dependencies=[Depends(verify_key)],
            response_class=ORJSONResponse,
            description="takes in a new entry for DB calculates the scores and updates the table"
        )
        def add_entry(add_term: AddInputModel):
            term: str = add_term.add_entry
            process_start = time.time()
            self.__searcher.add_entry(term)
            process_end = time.time()
            self.__logger.info(f"Time to add entry: {term}: {round((process_end-process_start), 3)} ms")
            return ORJSONResponse({"added_entry": f"{term}"})

        return api_router
