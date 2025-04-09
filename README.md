# Deep Research LangGraph

Deep Research LangGraph is an AI‑powered research assistant that automates web searches, content extraction, and report generation using LangGraph citeturn2view0, Google Gemini API citeturn8view0, and Streamlit citeturn4view0.

## Table of Contents

- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [Architecture](#architecture)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Configuration](#configuration)  
- [Usage](#usage)  
  - [Running the Streamlit App](#running-the-streamlit-app)  
  - [Using the `ResearchAgent` Class](#using-the-researchagent-class)  
- [Project Structure](#project-structure)  
- [Modules](#modules)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

## Features

- **Query Generation:** Generates diverse, facet‑focused search queries from a user prompt citeturn2view0  
- **Multi‑Source Search:** Searches DuckDuckGo and Wikipedia for each query citeturn7view0  
- **Content Extraction:** Scrapes and cleans HTML content from web pages and Wikipedia articles citeturn7view0  
- **Report Synthesis:** Uses Google Gemini API to produce a comprehensive, well‑structured report citeturn2view0  
- **Interactive UI:** Streamlit app with progress bars, tabs, and download buttons citeturn4view0  
- **Conversational Chat:** Chat interface powered by LangGraph for follow‑up questions citeturn2view0  

## Tech Stack

- **Python 3.8+** citeturn8view0  
- **Streamlit** for the web interface citeturn4view0  
- **LangGraph** for workflow orchestration citeturn2view0  
- **Google Generative AI** (`google-generativeai`) for LLM calls citeturn8view0  
- **DuckDuckGo Search** (`duckduckgo-search`) citeturn7view0  
- **Wikipedia API** (`wikipedia`) citeturn7view0  
- **BeautifulSoup** (`beautifulsoup4`) for HTML parsing citeturn7view0  
- **Pydantic** for data models citeturn2view0  
- **Loguru / Logging** for diagnostics citeturn6view0  

## Architecture

At its core, Deep Research LangGraph consists of:  
1. **`ResearchAgent`**: A LangGraph‐based state machine that orchestrates query generation, searches, content extraction, and report synthesis citeturn2view0  
2. **Tool Modules**: Independent functions for searching and scraping (`tools.py`) citeturn7view0  
3. **Streamlit App**: Provides a user interface, progress feedback, and download options (`app.py`) citeturn4view0  
4. **Logging**: Configured via `logger.py` for both console and file outputs citeturn6view0  

## Getting Started

### Prerequisites

- Python 3.8 or higher  
- A valid Google Gemini API key citeturn4view0  

### Installation

```bash
git clone https://github.com/Fayaz409/Deep_Research-LangGraph.git
cd Deep_Research-LangGraph/deep_search
pip install -r requirements.txt
```  
citeturn8view0

### Configuration

Create a `.env` file in the `deep_search/` directory:

```ini
GEMINI_API_KEY=your_google_gemini_api_key_here
```  
citeturn4view0

## Usage

### Running the Streamlit App

```bash
cd deep_search
streamlit run app.py
```  
Then open the URL shown in your browser. citeturn4view0

### Using the `ResearchAgent` Class

```python
from deep_search.agent import ResearchAgent

agent = ResearchAgent(api_key="YOUR_KEY_HERE")
result = agent.research("Quantum computing fundamentals")
print(result.report)
```  
citeturn2view0

## Project Structure

```
Deep_Research-LangGraph/
└── deep_search/
    ├── agent.py         # ResearchAgent implementation
    ├── tools.py         # Search & scraping tool functions
    ├── logger.py        # Logging configuration
    ├── app.py           # Streamlit web application
    └── requirements.txt # Project dependencies
```  
citeturn1view0

## Modules

- **`agent.py`**: Defines `ResearchAgent`, orchestrates the research workflow with LangGraph citeturn2view0  
- **`tools.py`**: Implements search (DuckDuckGo, Wikipedia) and HTML content extraction citeturn7view0  
- **`logger.py`**: Sets up console and file logging, with dynamic log‑level control citeturn6view0  
- **`app.py`**: Streamlit app with tabs for Research, Chat, and History citeturn4view0  

## Contributing

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m "Add YourFeature"`)  
4. Push to the branch (`git push origin feature/YourFeature`)  
5. Open a Pull Request  

Please follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and include tests where applicable.

## License

This project is currently unlicensed. To apply a license, add a `LICENSE` file to the repository.

## Contact

For questions or feedback, please open an issue or reach out to the project maintainer.
