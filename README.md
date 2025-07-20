# ScholarSynth AI - Literature Review Generator

A simple web application that automatically generates literature reviews using AI agents and arXiv papers.

## What it does

- Search for research papers on arXiv
- Generate comprehensive literature reviews automatically
- Download reviews as Markdown or text files

## Setup

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Get Groq API key**
- Sign up at [console.groq.com](https://console.groq.com/keys)
- Get your free API key

3. **Set API key**
```bash
export GROQ_API_KEY="your_api_key_here"
```

4. **Run the app**
```bash
streamlit run streamlit_app.py
```

## How to use

1. Enter a research topic (e.g., "machine learning")
2. Click "Generate Review"
3. Wait for the AI to create your literature review
4. Download the result

## Files

- `streamlit_app.py` - Main web interface
- `agent_be.py` - AI agents that do the work
- `groq_client.py` - Connects to Groq API
- `requirements.txt` - Python packages needed

## Requirements

- Python 3.8+
- Groq API key (free)
- Internet connection

## Technologies used

- **AutoGen** - Multi-agent framework
- **Groq API** - Fast AI models
- **Streamlit** - Web interface
- **arXiv** - Research papers database

## License

MIT License
