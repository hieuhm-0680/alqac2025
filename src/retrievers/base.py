
from abc import abstractmethod
from typing import List

from pydantic import BaseModel

from src.models.schemas import Document


class BaseRetriever(BaseModel):
    """Abstract base class for a Document retrieval system.

    A retrieval system is defined as something that can take string queries and return
    the most 'relevant' Documents from some source.
    
    When implementing a custom retriever, the class should implement
    the `_get_relevant_documents` method to define the logic for retrieving documents.
    """
    @abstractmethod
    def _get_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Subclasses must implement this method.")

    def retrieve(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

