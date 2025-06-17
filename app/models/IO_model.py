from typing import List

from pydantic import BaseModel, Field


class SearchInputModel(BaseModel):
    search_entry: str = Field(
            ...,
            description="Input field for search endpoint, takes in terms that need to be searched",
            examples=["Who won the 1992 NBA Championship?", "What are the best selling items of 2024"]
        )


class SearchOutputModel(BaseModel):
    document_indexes: List[str] = Field(
            ...,
            description="List of top 10 result indexes descending order",
            examples=["[50, 35, 80, 55, 66, 88, 22, 68, 32, 99]"]
        )


class AddInputModel(BaseModel):
    add_entry: str = Field(
            ...,
            description="Input field for Add endpoint, takes in terms that need to be added to our document DB",
            examples=["Who won the 1992 NBA Championship?", "What are the best selling items of 2024"]
        )


class RAGInputModel(BaseModel):
    question: str = Field(..., example="What did the 1992 NBA championship involve?")


class SearchResult(BaseModel):
    document_index: str = Field(..., description="Document ID or file name")
    document_text: str = Field(..., description="The full text of the retrieved document")


class RAGOutputModel(BaseModel):
    question: str
    retrieved_documents: List[SearchResult]
    simulated_answer: str = Field(..., description="Answer from the LLM (simulated or real)")

