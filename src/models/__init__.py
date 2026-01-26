"""
NovaVision - Emotion to Image AI Generator
Author: Urme Bose (urme-b)
License: MIT
Repository: https://github.com/urme-b/NovaVision

Models Package
==============
This package contains Pydantic models for data validation and serialization.

Exports:
    - EmotionType: Enumeration of supported emotion categories
    - EmotionResult: Emotion analysis output structure
    - ImageGenerationRequest: API request structure
    - GenerationResult: Complete pipeline output structure
"""

from src.models.schemas import EmotionType, EmotionResult, ImageGenerationRequest, GenerationResult

__all__ = ["EmotionType", "EmotionResult", "ImageGenerationRequest", "GenerationResult"]
