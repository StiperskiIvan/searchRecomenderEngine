from typing import List
import nltk
from logging import Logger


def download_nltk_model(logger: Logger, models: List[str] = None):
    if models is not None:
        for model in models:
            logger.info(f"Downloading NLTK model {model} from NLTK registry")
            try:
                nltk.download(model)
            except Exception as e:
                logger.error(f"Error downloading model {model} from registry. Error: {repr(e)}")
    else:
        logger.info("Nothing to download from NLTK registry")
