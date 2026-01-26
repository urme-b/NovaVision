"""
NovaVision - Emotion to Image AI Generator
Author: Urme Bose (urme-b)
License: MIT
Repository: https://github.com/urme-b/NovaVision

Application Settings Module
===========================
This module provides centralized configuration management for NovaVision using
Pydantic Settings. It loads configuration from environment variables and .env
files, providing type validation and default values.

Configuration Options:
    - HF_TOKEN: HuggingFace API token (required for image generation)
    - emotion_model: Model identifier for emotion classification
    - image_model: Model identifier for image generation
    - Generation parameters: num_images, guidance_scale, inference_steps

Environment Variables:
    HF_TOKEN - Required. Your HuggingFace API token for Inference API access.
               Get yours at: https://huggingface.co/settings/tokens

Usage:
    from config.settings import get_settings

    settings = get_settings()
    print(settings.app_name)  # "NovaVision"
    print(settings.hf_token)  # Your HF token from .env
"""

# Standard library imports
from functools import lru_cache  # Caching decorator for singleton pattern

# Third-party imports
from pydantic_settings import BaseSettings  # Pydantic settings management
from pydantic import Field  # Field validation and metadata


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    This class uses Pydantic Settings to automatically load configuration
    from environment variables and .env files. It provides type validation,
    default values, and easy access to all configuration options.

    Attributes:
        app_name: Application name for display purposes.
        app_description: Brief description of the application.
        hf_token: HuggingFace API token for model inference.
        emotion_model: HuggingFace model ID for emotion classification.
        image_model: HuggingFace model ID for image generation.
        default_num_images: Default number of images to generate per request.
        default_guidance_scale: Default guidance scale for diffusion models.
        default_num_inference_steps: Default number of denoising steps.

    Environment Variable Mapping:
        HF_TOKEN -> hf_token (required, no default)

    Example:
        >>> settings = Settings()
        >>> settings.app_name
        'NovaVision'
        >>> settings.hf_token
        'hf_xxxxxxxxxxxxx'  # From .env file
    """

    # Application Metadata
    # --------------------
    # Basic application information for display and identification
    app_name: str = "NovaVision"
    app_description: str = "Transform emotions into stunning AI-generated images"

    # HuggingFace Configuration
    # -------------------------
    # API token required for accessing HuggingFace Inference API
    # The 'alias' parameter maps to the HF_TOKEN environment variable
    hf_token: str = Field(..., alias="HF_TOKEN")

    # Model Configurations
    # --------------------
    # Default model identifiers for emotion analysis and image generation
    # These can be overridden via environment variables if needed
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    image_model: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # Generation Settings
    # -------------------
    # Default parameters for image generation
    # These provide sensible defaults that can be overridden per-request
    default_num_images: int = 1           # Number of images per generation
    default_guidance_scale: float = 7.5   # How closely to follow the prompt (1-20)
    default_num_inference_steps: int = 30 # Number of denoising steps (10-50)

    class Config:
        """
        Pydantic Settings configuration.

        Specifies how settings should be loaded from the environment:
        - env_file: Load variables from .env file in project root
        - env_file_encoding: UTF-8 encoding for .env file
        - extra: Ignore extra environment variables not defined in the model
        """
        env_file = ".env"          # Load from .env file
        env_file_encoding = "utf-8"  # UTF-8 encoding
        extra = "ignore"           # Ignore undefined environment variables


@lru_cache
def get_settings() -> Settings:
    """
    Get the cached settings instance.

    Uses LRU cache to ensure settings are only loaded once,
    implementing a singleton pattern for configuration access.

    Returns:
        Settings: The application settings instance.

    Example:
        >>> settings = get_settings()
        >>> print(settings.emotion_model)
        'j-hartmann/emotion-english-distilroberta-base'

    Note:
        The cache persists for the lifetime of the application.
        Changes to .env require an application restart.
    """
    return Settings()
