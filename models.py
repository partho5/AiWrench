from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class AssetContext(BaseModel):
    type: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    mileage: Optional[int] = None
    specs: Optional[dict[str, Any]] = None


class EnrichRequest(BaseModel):
    threadId: str
    messages: list[Message]
    assetContext: AssetContext
    toolbox: list[str] = Field(default_factory=list)
    serviceHistory: list[dict[str, Any]] = Field(default_factory=list)
    skillLevel: int = Field(default=5, ge=1, le=10)
    imageUrl: Optional[str] = None  # Cloudinary URL; server calls Grok Vision if present


class EnrichResponse(BaseModel):
    answer: str
    confidence: int
    confidence_reasons: list[str]
    abstain: bool
    safety_flag: bool
    role: str = "assistant"


class VisionRequest(BaseModel):
    imageUrl: str
    systemPrompt: Optional[str] = None
    assetContext: Optional[dict[str, Any]] = None


class StructuredFindings(BaseModel):
    part: Optional[str] = None
    condition: Optional[str] = None
    severity: Optional[str] = None
    observations: list[str] = Field(default_factory=list)


class VisionResponse(BaseModel):
    analysis: str
    confidence: int
    structured_findings: StructuredFindings


class ClassifyRequest(BaseModel):
    visionResult: Optional[str] = None
    textDescription: str
    assetId: str


class ClassifyResponse(BaseModel):
    assetSpecs: dict[str, Any]
    condition: str
    confidence: int
    recommendations: list[str]
    reminders: list[dict[str, Any]] = Field(default_factory=list)


class UserAnswer(BaseModel):
    q: str
    a: str


class RefineRequest(BaseModel):
    initialClassification: dict[str, Any]
    userAnswers: list[UserAnswer]
    skillLevel: int = Field(default=5, ge=1, le=10)


class RefineResponse(BaseModel):
    assetSpecs: dict[str, Any]
    condition: str
    confidence: int
    reminders: list[dict[str, Any]] = Field(default_factory=list)
    refined_recommendations: list[str] = Field(default_factory=list)
