import json
import logging
import os
import pickle
import re
import sys
import tarfile

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import Config
from app.external_downloader import download_nltk_model

# using only lematization (if further speed is necessary needed add stemming)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(name=Config.LOGGER_NAME)
download_nltk_model(logger=logger, models=["stopwords", "wordnet"])
INPUT_LANGUAGE = "english"
TAR_FILE_NAME = "20news-bydate.tar.gz"
OUTPUT_FILE_NAME = "cleaned_20newsgroups.json"
VECTORIZED_OUTPUT_FILE_NAME = "tfidf_data.pkl"
VECTORIZER_OUTPUT_FILE_NAME = "vectorizer.pkl"
LOWERCASE = False


def preprocess_text(file_name: str, text: str, language: str = INPUT_LANGUAGE) -> str:
    """
    Cleans, tokenizes, and lemmatizes text.
    - Removes special characters and digits.
    - Converts text to lowercase.
    - Removes stopwords.
    - Applies lemmatization.

    Args:
        file_name: file code(id)
        text: The input text to preprocess
        language: language of input text
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
    logger.info(f"File {file_name} cleaned and processed")

    return " ".join(cleaned_words)


def extract_and_process(tar_file_path: str, output_file_path: str):
    """
    Extracts the 20 Newsgroups dataset from a tar.gz file, preprocesses the text,
    and saves the cleaned dataset to a JSON file.

    Args:
        tar_file_path (str): Path to the 20 Newsgroups tar.gz file.
        output_file_path (str): Path to save the preprocessed JSON dataset.
    """
    # Extract the dataset
    with tarfile.open(tar_file_path, "r:gz") as tar:
        tar.extractall()
        extracted_dir = tar.getnames()[0].split('/')[0]

    dataset = []
    for root, _, files in os.walk(extracted_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, "r", encoding="latin1") as file:
                    content = file.read()

                # Preprocess the content
                preprocessed_content = preprocess_text(text=content, file_name=file_name)

                # Add to the dataset
                dataset.append({
                    "file_name": file_name,
                    "original_text": content,  # Optional: Keep original text
                    "processed_text": preprocessed_content
                })
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {repr(e)}")

    # Save the dataset to a JSON file
    with open(output_file_path, "w", encoding="utf-8") as json_file:
        json.dump(dataset, json_file, indent=4, ensure_ascii=False)

    logger.info(f"Processed dataset saved to {output_file_path}")

    return dataset


def load_preprocessed_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# Define a function to vectorize the data
def vectorize_documents(documents):
    # Extract only the text content from the preprocessed dataset
    corpus = [doc['processed_text'] for doc in documents]

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(lowercase=LOWERCASE)

    # Fit the vectorizer on the corpus
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")  # (n_documents, n_features)
    return tfidf_matrix, feature_names, vectorizer


def save_tfidf_vectors_pickle(tfidf_matrix, documents, feature_names, vectorizer, vectorizer_file, output_file):
    """
    Save TF-IDF vectors along with document metadata to a Pickle file.

    :param tfidf_matrix: The TF-IDF matrix (sparse).
    :param documents: The list of document metadata (e.g., file_name, original_text).
    :param feature_names: List of feature names from the vectorizer.
    :param vectorizer: Vectorizer calculated
    :param vectorizer_file: Vectorizer calculated path
    :param output_file: Path to save the Pickle file.
    """
    # Create a structure to store all necessary information
    results = []
    for i, doc in enumerate(documents):
        # Get non-zero TF-IDF scores for each document
        row = tfidf_matrix.getrow(i)
        tfidf_vector = {feature_names[j]: row[0, j] for j in row.indices}
        results.append({
            "file_name": doc["file_name"],
            "tfidf_vector": tfidf_vector
        })

    # Save results to a Pickle file
    with open(output_file, "wb") as pickle_file:
        pickle.dump(results, pickle_file)

        # Save the vectorizer to a Pickle file
    with open(vectorizer_file, "wb") as vectorizer_pickle:
        pickle.dump(vectorizer, vectorizer_pickle)
    logger.info(f"Vectorizer saved to {vectorizer_file}")


if __name__ == "__main__":

    current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tar_file_path = os.path.join(current_path, "app", "data", "preprocessed_data", TAR_FILE_NAME)
    output_file_path = os.path.join(current_path, "app", "data", "proccessed_data", OUTPUT_FILE_NAME)
    output_file_path_vectorized = os.path.join(current_path, "app", "data", "proccessed_data", VECTORIZED_OUTPUT_FILE_NAME)
    output_file_path_vectorizer = os.path.join(current_path, "app", "data", "proccessed_data", VECTORIZER_OUTPUT_FILE_NAME)
    # Process the dataset
    cleaned_data = extract_and_process(tar_file_path, output_file_path)
    # cleaned_data = load_preprocessed_data(output_file_path)
    # Calculate weights for TF-IDF matrix
    tfidf_matrix, feature_names, vectorizer = vectorize_documents(cleaned_data)
    save_tfidf_vectors_pickle(tfidf_matrix, cleaned_data, feature_names, vectorizer, output_file_path_vectorizer, output_file_path_vectorized)
