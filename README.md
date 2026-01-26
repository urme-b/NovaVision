# NovaVision

**Transform Emotions into AI-Generated Art**

NovaVision is an intelligent emotion-to-image application that analyzes text input to detect emotional content and generates stunning AI artwork that visually represents those emotions. It combines state-of-the-art NLP for emotion detection with cutting-edge diffusion models for image generation.

---

## Features

### Live Emotion Analysis
Real-time NLP-powered emotion detection as you type using a fine-tuned DistilRoBERTa model:
- Detects 7 emotions: Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral
- Displays confidence scores with animated progress bars
- Shows valence (positive/negative) and arousal (energy level) metrics

### Smart Prompt Engineering
Intelligent detection system that differentiates between:
- **Emotional inputs** ("I feel peaceful") - Generates mood-inspired scenic artwork
- **Object/scene inputs** ("A majestic lion") - Generates exactly what you describe
- Automatically enhances prompts with quality modifiers and anti-watermark tokens

### AI Image Generation
High-quality image generation using FLUX.1 diffusion models:
- **FLUX.1-dev** for premium quality (30 inference steps)
- Automatic fallback to **FLUX.1-schnell** for faster results
- 1024x1024 resolution output
- 5 style presets: Photorealistic, Artistic, Nature, Abstract, Dreamscape

### Additional Features
- **Download Options**: Save images as PNG or full analysis reports
- **Seed Control**: Reproducible results with specific seeds
- **Image History**: Gallery of last 4 generated images
- **Prompt Display**: See how your input is transformed and enhanced

---

## Tech Stack

| Technology | Purpose | Details |
|------------|---------|---------|
| **Python 3.9+** | Backend runtime | Core application language |
| **Flask** | Web server | RESTful API endpoints |
| **Gradio** | Alternative UI | ML-focused web interface |
| **HuggingFace Transformers** | NLP | Emotion classification models |
| **HuggingFace Inference API** | Image Generation | Cloud-based model serving |
| **FLUX.1-dev** | Diffusion Model | High-quality image generation |
| **DistilRoBERTa** | Emotion Model | j-hartmann fine-tuned model |
| **Pydantic** | Validation | Data models and settings |
| **Pillow** | Image Processing | Image handling and conversion |

---

## Architecture

```
NovaVision/
├── server.py                    # Flask server & API endpoints
├── app.py                       # Gradio alternative interface
├── index.html                   # Frontend SPA (HTML/CSS/JS)
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (HF_TOKEN)
├── config/
│   ├── __init__.py
│   └── settings.py              # Pydantic settings management
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic data models
│   └── services/
│       ├── __init__.py
│       ├── emotion_analyzer.py  # NLP emotion detection
│       ├── image_generator.py   # FLUX.1 image generation
│       └── pipeline.py          # End-to-end orchestration
├── tests/
│   ├── __init__.py
│   └── test_emotion_analyzer.py # Unit tests
└── outputs/                     # Generated images
```

### Data Flow

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ User Input  │───▶│ Emotion Analysis │───▶│ Smart Detection │
│   (Text)    │    │  (DistilRoBERTa) │    │ (Emotion/Object)│
└─────────────┘    └──────────────────┘    └────────┬────────┘
                                                    │
┌─────────────┐    ┌──────────────────┐    ┌────────▼────────┐
│   Display   │◀───│ FLUX.1 API Call  │◀───│ Prompt Building │
│   Result    │    │  (HuggingFace)   │    │ (Style+Quality) │
└─────────────┘    └──────────────────┘    └─────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- HuggingFace account with API token
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/urme-b/NovaVision.git
   cd NovaVision
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file with your HuggingFace token
   echo "HF_TOKEN=your_huggingface_token_here" > .env
   ```
   Get your token from: https://huggingface.co/settings/tokens

5. **Run the application**
   ```bash
   python server.py
   ```

6. **Open in browser**
   ```
   http://localhost:8000
   ```

---

## Usage

1. **Enter your text**: Describe how you're feeling or what you want to see
2. **Select a style**: Choose from Artistic, Nature, Abstract, Dreamscape, or Photorealistic
3. **Click Generate**: Watch as your emotions transform into art
4. **Explore results**: View emotion analysis, download images, or regenerate

### Example Inputs

| Input Type | Example | Result |
|------------|---------|--------|
| Emotional | "I feel peaceful watching the sunset" | Serene landscape inspired by tranquility |
| Emotional | "Excited about starting a new adventure" | Dynamic, energetic imagery |
| Object | "A majestic lion in the savanna" | Photorealistic lion image |
| Scene | "Cozy coffee shop on a rainy day" | Detailed coffee shop scene |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve main HTML interface |
| `/api/analyze` | POST | Real-time emotion analysis |
| `/api/generate` | POST | Full generation pipeline |

### Example API Call

```bash
# Emotion Analysis
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel happy today!"}'

# Image Generation
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "A peaceful morning", "style": "nature"}'
```

---

## Screenshots

<details>
<summary>Click to view screenshots</summary>

### Main Interface
*Screenshot placeholder - Main generation interface*

### Live Emotion Analysis
*Screenshot placeholder - Real-time emotion detection*

### Generated Artwork
*Screenshot placeholder - Sample generated images*

</details>

---

## Skills Demonstrated

This project showcases expertise in:

| Skill | Implementation |
|-------|----------------|
| **Machine Learning Integration** | HuggingFace Transformers for emotion classification |
| **API Development** | RESTful API design with Flask |
| **Prompt Engineering** | Advanced techniques for image generation quality |
| **Full-Stack Development** | End-to-end application with Python backend + JS frontend |
| **Modern Python** | Type hints, dataclasses, Pydantic models |
| **Software Architecture** | Clean separation of concerns, service-based design |
| **UX Design** | Intuitive interface with loading states and error handling |
| **Data Validation** | Pydantic schemas for type safety |

---

## Testing

Run the test suite to verify all components work correctly:

```bash
# Install pytest (if not already installed)
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_services.py -v
```

### Test Coverage
- **EmotionAnalyzer**: 8 tests (emotion detection, valence, arousal, confidence)
- **ImageGenerator**: 2 tests (smart detection for emotional vs object inputs)

---

## Future Improvements

- [ ] Add more style presets and customization options
- [ ] Implement image-to-image generation for style transfer
- [ ] Add user authentication and generation history
- [ ] Deploy to HuggingFace Spaces for public demo
- [ ] Add batch processing for multiple generations
- [ ] Implement caching for faster subsequent generations

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Author

**Urme Bose**

- GitHub: [@urme-b](https://github.com/urme-b)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [HuggingFace](https://huggingface.co/) for transformer models and inference API
- [Black Forest Labs](https://blackforestlabs.ai/) for FLUX.1 models
- [j-hartmann](https://huggingface.co/j-hartmann) for the emotion classification model

---

<p align="center">
  <strong>NovaVision</strong> - Where emotions become art
</p>
