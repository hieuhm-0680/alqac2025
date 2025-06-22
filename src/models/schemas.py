"""Data models and schemas for ALQAC 2025."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class QuestionType(str, Enum):
    """Types of questions in the competition."""

    TRUE_FALSE = "Đúng/Sai"
    MULTIPLE_CHOICE = "Trắc nghiệm"
    FREE_TEXT = "Tự luận"


class TaskType(str, Enum):
    """Types of tasks in the competition."""

    RETRIEVAL = "retrieval"
    QA = "qa"


class Document(BaseModel):
    """Legal document model."""

    law_id: str = Field(..., description="Law identifier")
    article_id: str = Field(..., description="Article identifier")
    text: str = Field(..., description="Full text of the article")

class QuestionBase(BaseModel):
    """Base question model."""

    question_id: str = Field(..., description="Unique question identifier")
    question_type: QuestionType = Field(..., description="Type of question")
    text: str = Field(..., description="Question text")


class TrainingQuestion(QuestionBase):
    """Training question with answer and relevant articles."""

    relevant_documents: List[Document] = Field(
        ..., description="List of relevant legal articles"
    )
    choices: Optional[Dict[str, str]] = Field(
        None, description="Choices for multiple choice questions"
    )
    answer: str = Field(..., description="Correct answer")

    @field_validator("choices")
    def validate_choices(
        cls, v: Optional[Dict[str, str]], values: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """Validate choices for multiple choice questions."""
        question_type = values.get("question_type")
        if question_type == QuestionType.MULTIPLE_CHOICE and not v:
            raise ValueError("Multiple choice questions must have choices")
        elif question_type != QuestionType.MULTIPLE_CHOICE and v:
            raise ValueError("Only multiple choice questions can have choices")
        return v

    @field_validator("answer")
    def validate_answer(cls, v: str, values: Dict[str, Any]) -> str:
        """Validate answer format based on question type."""
        question_type = values.get("question_type")

        if question_type == QuestionType.TRUE_FALSE:
            if v not in ["Đúng", "Sai"]:
                raise ValueError("True/False answers must be 'Đúng' or 'Sai'")
        elif question_type == QuestionType.MULTIPLE_CHOICE:
            if v not in ["A", "B", "C", "D"]:
                raise ValueError(
                    "Multiple choice answers must be A, B, C, or D")

        return v


class TestQuestion(QuestionBase):
    """Test question without answer (for evaluation)."""

    choices: Optional[Dict[str, str]] = Field(
        None, description="Choices for multiple choice questions"
    )


class RetrievalResult(BaseModel):
    """Result for document retrieval task."""

    question_id: str = Field(..., description="Question identifier")
    relevant_articles: List[Document] = Field(
        ..., description="Retrieved relevant articles"
    )


class QAResult(BaseModel):
    """Result for question answering task."""

    question_id: str = Field(..., description="Question identifier")
    answer: str = Field(..., description="Predicted answer")


# Considering...
class EvaluationMetrics(BaseModel):
    """Evaluation metrics for the competition."""

    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f2_score: float = Field(..., ge=0, le=1)
    accuracy: Optional[float] = Field(None, ge=0, le=1)


class ModelPrediction(BaseModel):
    """Model prediction with confidence scores."""

    prediction: Union[str, List[Document]]
    confidence: float = Field(..., ge=0, le=1)
    reasoning: Optional[str] = Field(None, description="Model reasoning")
