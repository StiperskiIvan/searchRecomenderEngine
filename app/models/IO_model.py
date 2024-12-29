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
