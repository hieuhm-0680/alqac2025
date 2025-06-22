
from pydantic import BaseModel


class BaseAnswerer(BaseModel):
    """
    Base class for all answerers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def answer(self, question: str) -> str:
        """
        Answer the question.
        """
        raise NotImplementedError("Subclasses must implement this method.")