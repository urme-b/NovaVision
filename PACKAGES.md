# Package Structure

```
NovaVision/
├── app.py                    # Main application
├── server.py                 # API server
├── config/
│   └── settings.py           # Configuration
├── src/
│   ├── services/
│   │   ├── emotion_analyzer.py   # NLP emotion detection
│   │   ├── image_generator.py    # FLUX.1 integration
│   │   └── pipeline.py           # Orchestration
│   └── models/
│       └── schemas.py            # Data models
├── index.html                # Frontend
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── LIBRARIES.md              # Library docs
├── PACKAGES.md               # This file
└── ARCHITECTURE.md           # System design
```

## Key Modules

**EmotionAnalyzer** - Detects 7 emotions (joy, sadness, anger, fear, surprise, disgust, neutral) with confidence scores

**ImageGenerator** - Generates images via FLUX.1 with smart detection (emotion vs object mode)

**PromptBuilder** - Enhances prompts with style modifiers and quality keywords
