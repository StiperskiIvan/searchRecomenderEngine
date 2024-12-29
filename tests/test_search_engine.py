import logging
from typing import List, Dict

import pytest
from pydantic import ValidationError

from app.config import Config
from app.models.IO_model import SearchOutputModel
from app.services.search_engine import SearchEngine
from app.external_downloader import download_nltk_model

logger = logging.getLogger(Config.LOGGER_NAME)
download_nltk_model(logger=logger, models=["stopwords", "wordnet"])


def test_search():
    engine = SearchEngine(logger=logger)
    input_query = "Is this thing working"
    expected_output: List[str] = ['76117', '105030', '84435', '52293', '54077', '54058', '21713', '10867', '178719', '84193']
    results = engine.search_tfidf(query=input_query)
    engine.add_entry(input_query)
    assert results == expected_output, f"Expected {expected_output} but got {results}"


def test_validate():
    engine = SearchEngine(logger=logger)
    input_query = "Is this thing working"
    expected_output: List[str] = ['76117', '105030', '84435', '52293', '54077', '54058', '21713', '10867', '178719', '84193']
    try:
        results = engine.search_tfidf(query=input_query)
        results_formatted: Dict[str: List[str]] = {"document_indexes": results}
        SearchOutputModel.model_validate(results_formatted)
        assert results == expected_output, f"Expected {expected_output} but got {results}"
    except ValidationError as e:
        pytest.fail(f"Pydantic validation failed: {e}")


