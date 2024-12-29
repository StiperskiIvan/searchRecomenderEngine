# Ranking Search Engine App

## Overview

This application is built using FastAPI and provides a set of RESTful API endpoints for various services. 
The app includes functionality for searching, adding entries, and checking the system status. 
User input is taken and a TF-IDF scoring method is used to search the 20newsgroup dataset, 
and the top 10 similar document indexes are returned. User can also add new entries to the database.
Database is cleaned, lematized, and the texts were stripped of stopwords and additional punctuations.
A TF-IDF matrix with scores, and the vectorizer were calculated from all the dataset, and stored beforehand.
A script to repeat those steps can be found in `prepare_data` folder.

## Features

- **Search Endpoint**: Allows users to search for terms and retrieve the top 10 most similar results using the TF-IDF method.
- **Add Entry Endpoint**: Lets users add new entries to the database and recalculates the scores for documents.
- **Status Endpoint**: Provides the current status of the API, including health checks.

## Installation

### Prerequisites

Before setting up the application, make sure you have the following installed:

- Python > 3.9
- Docker (if using containerization)
- Uvicorn (for serving FastAPI)

### Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/StiperskiIvan/searchRecomenderEngine/
    cd <application/root/directory>
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. If you want to run the app locally:

    ```bash
    uvicorn app.main:app --reload
    ```

    This will start the FastAPI app on `http://localhost:8000`.

## Docker Setup

If you'd prefer to run the app in a Docker container, follow the steps below:

1. Build the Docker image:

    ```bash
    docker build -t ranking-search-engine-app . --progress=plain
    ```
   This `--progress=plain` allows you to see docker building steps, it is Optional. Docker build also runs tests before building to ensure the app is working properly


3. Run the Docker container:

    ```bash
    docker run -p 8000:8000 ranking-search-engine-app 
    ```


The app will be accessible at `http://localhost:8000` on your local machine.

## API Endpoints

Sample requests can be found in Example `requests.txt`

### `/ranking-engine/status` (GET)

This endpoint returns the current status of the API.

#### Response:

```json
{
  "status": 200,
  "ok": true
}
```

### `/ranking-engine/search` (POST)

This endpoint allows searching for a term in the document database and returns indices of top 10 similar documents

#### Payload:

```json
{
  "search_entry": "sample search term"
}
```

#### Response:

```json
{
  "document_indexes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
```

### `/ranking-engine/add_entry ` (POST)

This endpoint adds a new entry to the document database. By default, parameter `UPDATE_TD_MATRIX` is switched to False which speeds up the entry time, but the term is not instantly searchable after since the matrix and the vector are not recalculated. This is a time costly operation so that update should be ideally handled differently (eg. periodic batch update) 

#### Payload:
```json
{
  "add_entry": "Who won the 1992 NBA Championship?"
}
```

### Response:
```json
{
  "added_entry": "Who won the 1992 NBA Championship?"
}
```

## Testing

Perquisites: Before running the tests make sure to have `pytest` library installed:

```bash
pip install pytest
```

Navigate to root folder of this repo and run: 
```bash
pytest tests/test_search_engine.py
```

## Additional improvements

To further improve latency of the service especially, search endpoint a skeleton of that implementation is nested in the project. When receiving a search term a Redis DB can be used as a caching layer. First the original term can be checked if it already stored in Redis before doing the calculation.
Redis data would contain hash of the original input (to reduce size and contain uniqueness) and the value would be the search result (top 10 indexes) which would reduce the calculation cost and search time significantly.
Data would be stored with a TTL (time to live) set to reduce the storage of newer used search terms.

### Proposed Workflow for search endpoint:

- Hash the input term
- via `redis_client.get("term")` check if results are in redis if they are return the result
- if they are not go to calculation
- after calculation store the hashed input and the results in redis via `redis_client.set(key, value, ex=TTL)`
