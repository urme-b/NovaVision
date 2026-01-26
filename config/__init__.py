"""
NovaVision - Emotion to Image AI Generator
Author: Urme Bose (urme-b)
License: MIT
Repository: https://github.com/urme-b/NovaVision

Configuration Package
=====================
This package contains application configuration and settings management.

Exports:
    - Settings: Pydantic settings class for environment configuration
    - get_settings: Factory function for cached settings instance
"""

from config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
