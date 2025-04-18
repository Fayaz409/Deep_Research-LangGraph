# Deep Research LangGraph

Deep Research LangGraph is an AI-powered research assistant designed to automate the process of web searching, extracting relevant information, and synthesizing comprehensive reports. It leverages LangGraph for workflow orchestration, the Google Gemini API for generative capabilities, and Streamlit for an interactive user interface.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
  - [Configuration (API Key)](#configuration-api-key)
- [Usage](#usage)
  - [Running the Streamlit App](#running-the-streamlit-app)
  - [Using the `ResearchAgent` Class Programmatically](#using-the-researchagent-class-programmatically)
- [Modules](#modules)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This tool takes a user's research prompt, generates multiple focused search queries, searches across different sources (like DuckDuckGo and Wikipedia), extracts content from the results, and uses a Google Gemini model to generate a structured research report. It also provides a chat interface for follow-up questions based on the research context.

## Features

-   **Automated Query Generation:** Creates diverse search queries based on the initial research topic.
-   **Multi-Source Information Gathering:** Fetches data from DuckDuckGo web searches and Wikipedia articles.
-   **Web Content Extraction:** Intelligently scrapes and cleans relevant text content from HTML web pages.
-   **AI-Powered Report Synthesis:** Utilizes a Google Gemini model to analyze gathered information and generate a comprehensive report.
-   **Interactive Web UI:** A user-friendly Streamlit application provides input fields, progress visualization, results display (including the final report and sources), and download options.
-   **Conversational Follow-up:** Includes a chat interface (powered by LangGraph) to ask further questions related to the generated report.

## Tech Stack

-   **Python:** 3.8 or later
-   **Streamlit:** For building the interactive web application frontend.
-   **LangGraph:** For defining and running the stateful, multi-step research agent.
-   **Google Generative AI (`google-generativeai`):** To interact with Google Gemini language models.
-   **DuckDuckGo Search (`duckduckgo-search`):** For performing web searches.
-   **Wikipedia API (`wikipedia`):** For fetching Wikipedia article content.
-   **BeautifulSoup (`beautifulsoup4`):** For parsing and extracting content from HTML.
-   **Pydantic:** For data validation and settings management.
-   **Loguru / Logging:** For application logging and diagnostics.

## Architecture

The system is built around these core components:

1.  **`ResearchAgent` (`agent.py`):** A LangGraph state machine orchestrating the entire research process: query generation -> searching -> content scraping -> report generation.
2.  **Tooling (`tools.py`):** Contains functions for performing specific actions like web searches (DuckDuckGo), Wikipedia lookups, and HTML content scraping.
3.  **Streamlit Application (`app.py`):** The user interface that takes input, triggers the `ResearchAgent`, displays progress and results, and handles API key input via the sidebar.
4.  **Logging (`logger.py`):** Configures logging for monitoring and debugging purposes.

## Project Structure

```
Deep_Research-LangGraph/
└── deep_search/            # Main application package
    ├── __pycache__/        # Python cache files (usually ignored)
    ├── agent.py            # Contains the ResearchAgent LangGraph implementation
    ├── app.py              # The Streamlit web application script
    ├── logger.py           # Logging setup configuration
    ├── requirements.txt    # List of Python dependencies
    ├── tools.py            # Search and scraping tool functions
    └── README.md           # This file (should ideally be in the root folder)
```

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

-   **Python:** Ensure you have Python 3.8 or a newer version installed. You can check with `python --version`.
-   **Git:** Required for cloning the repository.
-   **Google Gemini API Key:** You need a valid API key from Google AI Studio. You can get one here: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Fayaz409/Deep_Research-LangGraph.git
    cd Deep_Research-LangGraph
    ```

2.  **Navigate to Project Directory:**
    All core files seem to be inside the `deep_search` folder based on your description and image.
    ```bash
    cd deep_search
    ```

3.  **Create and Activate a Virtual Environment (Recommended):**
    Using a virtual environment prevents dependency conflicts with other projects.
    * **On macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * **On Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    You should see `(venv)` prepended to your command prompt.

4.  **Install Dependencies:**
    Install all required libraries listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

### Configuration (API Key)

You need to provide your Google Gemini API key for the application to work. You have two primary options:

1.  **Via Streamlit Sidebar (Recommended):**
    * When you run the Streamlit application (`streamlit run app.py`), the sidebar will contain a text input field labeled "Enter your Gemini API Key".
    * Paste your API key there. The application will use it for the current session. This is convenient and avoids storing the key directly in the code.

2.  **Directly in `agent.py` (Alternative):**
    * Open the `deep_search/agent.py` file in a text editor.
    * Locate the section where the Gemini client is initialized or where the API key is expected. Look for a placeholder like `api_key="YOUR_KEY_HERE"` or `os.getenv("GEMINI_API_KEY")`.
    * Replace the placeholder or modify the code to directly include your API key as a string:
        ```python
        # Example modification in agent.py
        genai.configure(api_key="YOUR_ACTUAL_GEMINI_API_KEY_HERE")
        # or if passing to a class
        # agent = ResearchAgent(api_key="YOUR_ACTUAL_GEMINI_API_KEY_HERE")
        ```
    * **Caution:** Be careful not to commit your API key directly into version control (like Git) if you use this method.

**Gemini Model Selection:**
You can typically choose which Gemini model to use (e.g., `gemini-pro`, `gemini-1.5-pro-latest`, `gemini-1.0-pro`). This configuration is usually done within `agent.py` where the `GenerativeModel` is instantiated. Look for `genai.GenerativeModel('model-name')` and change `'model-name'` to your desired model.

## Usage

### Running the Streamlit App

1.  **Ensure your virtual environment is active** (you should see `(venv)` in your prompt).
2.  **Make sure you are in the correct directory:** `Deep_Research-LangGraph/deep_search/`.
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
4.  Streamlit will provide a local URL (usually `http://localhost:8501`). Open this URL in your web browser.
5.  If you haven't hardcoded the API key, enter your Google Gemini API Key in the sidebar input field.
6.  Enter your research topic in the main input area and start the research process.

### Using the `ResearchAgent` Class Programmatically

If you want to use the research capabilities directly within another Python script:

```python
from deep_search.agent import ResearchAgent # Adjust import path if needed

# Instantiate the agent, potentially passing the API key if not configured elsewhere
# Ensure you handle API key loading securely (e.g., environment variables)
# Option 1: Key provided directly
# agent = ResearchAgent(api_key="YOUR_GEMINI_API_KEY")

# Option 2: Key loaded from environment or sidebar internally (depends on agent.py setup)
agent = ResearchAgent() # Assuming agent handles key loading

# Define your research topic
topic = "Explain the basics of quantum entanglement for beginners"

# Run the research process
# The 'research' method likely takes the topic and returns the results
# The exact method signature might vary based on agent.py implementation
research_results = agent.research(topic)

# Access the generated report
print("Generated Report:")
print(research_results.report) # Accessing the 'report' attribute, adjust if needed

# Access other potential results like sources, queries, etc.
# print(research_results.sources)
```

## Modules

-   **`agent.py`**: Defines the core `ResearchAgent` class using LangGraph to manage the state and flow of the research process. Initializes connections to LLMs.
-   **`tools.py`**: Contains standalone functions for interacting with external services (DuckDuckGo Search, Wikipedia) and for processing web content (HTML scraping).
-   **`logger.py`**: Configures the logging system (e.g., Loguru) for outputting informational messages and errors to the console and/or files.
-   **`app.py`**: Implements the Streamlit user interface, handling user input, displaying progress and results, and managing API key entry through the sidebar.

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3.  Make your changes, adhering to code style (e.g., PEP 8).
4.  Add tests for your changes if applicable.
5.  Commit your changes (`git commit -m "Add concise description of change"`).
6.  Push to your branch (`git push origin feature/your-feature-name`).
7.  Open a Pull Request against the main repository branch.


## Contact

For questions, bug reports, or feedback, please open an issue on the GitHub repository.
