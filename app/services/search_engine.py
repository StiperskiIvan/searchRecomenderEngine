import os
import pickle
import re
from logging import Logger
from typing import List

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import lil_matrix
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity

from app.config import Config

INPUT_LANGUAGE = "english"
VECTORIZED_OUTPUT_FILE_NAME = "tfidf_data.pkl"
VECTORIZER_OUTPUT_FILE_NAME = "vectorizer.pkl"
TOP_K_RESULTS = Config.NO_OF_RESULTS_RETURNED

current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_file_path = os.path.join(current_path, "data", "proccessed_data", VECTORIZED_OUTPUT_FILE_NAME)
vectorizer_file_path = os.path.join(current_path, "data", "proccessed_data", VECTORIZER_OUTPUT_FILE_NAME)


def preprocess_text(text: str, language: str = INPUT_LANGUAGE) -> str:
    """
    Cleans, tokenizes, and lemmatizes text.
    - Removes special characters and digits.
    - Converts text to lowercase.
    - Removes stopwords.
    - Applies lemmatization.

    Args:
        :param text: The input text to preprocess
        :param language: language of input text
    Returns:
        str: The preprocessed and cleaned text.
    """

    # Remove common email headers like "From:", "Subject:", "Date:", etc. (Only at the beginning)
    text = re.sub(r"(?m)^(From:|Subject:|Date:|To:|Reply-To:|X-.*|Content-Type:|MIME-Version:).*", "", text)
    # Remove reply markers "Re:" only if it is part of the subject or the first line of the body
    text = re.sub(r"(^|\n)(Re:)\s*", "\n", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Remove special characters and digits
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.lower()

    # Tokenize and process
    stop_words = set(stopwords.words(language))
    lemmatizer = WordNetLemmatizer()

    # Tokenize and lemmatize
    words = text.split()
    cleaned_words = [
        lemmatizer.lemmatize(word) for word in words if word not in stop_words
    ]

    return " ".join(cleaned_words)


class SearchEngine:
    def __init__(self, logger: Logger = None) -> None:
        self.__logger = logger
        self.__model, self.__vectorizer = self.__load_model(
            model_path=model_file_path,
            vectorizer_path=vectorizer_file_path
        )
        self.__doc_matrix, self.__file_names = self.__prepare_document_vectors(
            vectorized_data=self.__model,
            vectorizer=self.__vectorizer
        )

    def __get_max_id(self):
        """
        Retrieve the maximum existing ID from the current dataset.
        """
        # Retrieve the highest ID from the existing file names
        try:
            max_id: int = max([int(file_name) for file_name in self.__file_names if file_name.isdigit()], default=0)
        except ValueError as e:
            self.__logger.error(f"Error retrieving max id while trying to add new data. Error: {repr(e)}")
            raise e
        return max_id

    def __load_model(self, model_path, vectorizer_path):
        try:
            with open(model_path, "rb") as pickle_file:
                tfidf_vectors = pickle.load(pickle_file)
            with open(vectorizer_path, "rb") as pickle_file:
                vectorizer = pickle.load(pickle_file)
        except Exception as e:
            self.__logger.error(f"Error loading models: {repr(e)}")
            raise e
        return tfidf_vectors, vectorizer

    def __prepare_document_vectors(self, vectorized_data, vectorizer):
        """
        Prepare document vectors from vectorized data in a sparse format.

        :param vectorized_data: List of dictionaries containing file_name and tfidf_vector.
        :param vectorizer: Fitted TF-IDF vectorizer with vocabulary.
        :return: (sparse_matrix, file_names), where sparse_matrix is a document-term matrix.
        """
        try:
            vocab_size = len(vectorizer.get_feature_names_out())
            num_docs = len(vectorized_data)

            # Create a sparse matrix in LIL format for efficient construction
            doc_matrix = lil_matrix((num_docs, vocab_size))
            file_names = []

            # Map terms to their indices in the vocabulary
            term_to_index = vectorizer.vocabulary_

            for doc_idx, entry in enumerate(vectorized_data):
                file_names.append(entry["file_name"])
                for term, value in entry["tfidf_vector"].items():
                    idx = term_to_index.get(term)
                    if idx is not None:
                        doc_matrix[doc_idx, idx] = value

        except Exception as e:
            self.__logger.error(f"Error preparing document vectors: {repr(e)}")
            raise e
        return doc_matrix.tocsr(), file_names

    def search_tfidf(self, query: str) -> List[str]:
        """
        Search for the top K most similar documents using cosine similarity.

        :param query: Query string to search.
        :return: List of top K document matches (file_name [indexes]).
        """
        # Preprocess query
        preprocessed_query = preprocess_text(query)

        # Vectorize the query
        query_vector = self.__vectorizer.transform([preprocessed_query])

        # Compute cosine similarities
        similarities = cosine_similarity(query_vector, self.__doc_matrix)[0]

        # Get top K results
        top_indices = similarities.argsort()[::-1][:TOP_K_RESULTS]
        results: List[str] = [str(self.__file_names[i]) for i in top_indices]

        return results

    def add_entry(self, text: str):
        """
        Add a new entry to the dataset.

        This method processes the new entry and adds its vectorized representation to
        the document matrix.

        :param text: The text content of the new entry.
        """
        # Preprocess the text
        preprocessed_text = preprocess_text(text=text)

        # Transform the new entry using the existing vectorizer
        new_vector = self.__vectorizer.transform([preprocessed_text])

        # Ensure the new vector's shape matches the document matrix
        if new_vector.shape[1] != self.__doc_matrix.shape[1]:
            raise ValueError(
                f"New vector dimensions {new_vector.shape[1]} do not match the document matrix dimensions {self.__doc_matrix.shape[1]}")

        # Stack the sparse matrices (new vector and document matrix) efficiently
        self.__doc_matrix = vstack([self.__doc_matrix, new_vector])

        # Add the new file name to the list of file names
        new_file_name: int = self.__get_max_id() + 1
        new_file_name: str = str(new_file_name)
        self.__file_names.append(new_file_name)

        """
        Optional: To update the TF-IDF model with the new entry, re-fit the vectorizer
        This is optional and computationally expensive, especially for large datasets
        Some kind of midterm storing and periodical batch scheduled update would be much better
        Can be updated to true if you want to include it
        """
        if Config.UPDATE_TD_MATRIX:
            self.__model, self.__vectorizer = self.__load_model(model_file_path, vectorizer_file_path)

        # Log the addition
        self.__logger.info(f"New entry added: {new_file_name}")

