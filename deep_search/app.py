"""
Streamlit web application for the AI Research Assistant.
"""
import streamlit as st
import os
import time
import tempfile
import subprocess
from dotenv import load_dotenv
import google.generativeai as genai
from agent import ResearchAgent
from logger import logger, set_log_level

# --- Initial Setup ---
st.set_page_config(
    page_title="AI Research Assistant",
    layout="wide",
    page_icon="🔍"
)

# Custom CSS styling for better UI
st.markdown("""
<style>
    /* Custom font imports */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Headers styling */
    h1, h2, h3, h4 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #1e3a8a;
    }
    
    /* Main title styling */
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.5rem 0;
    }
    
    /* Section headers */
    h2 {
        font-size: 1.8rem !important;
        border-bottom: 2px solid #ddd;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem !important;
    }
    
    /* Subsection headers */
    h3 {
        font-size: 1.4rem !important;
        margin-top: 1.2rem !important;
        color: #2563eb;
    }
    
    /* Block styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1e40af;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
        transform: translateY(-2px);
    }
    
    /* Form styling */
    .stForm {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 6px !important;
        border: 1px solid #cbd5e1 !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: #f1f5f9;
        border-radius: 6px 6px 0 0;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        font-weight: 500;
        color: #334155;
        background-color: #f8fafc;
        border-radius: 6px;
        padding: 0.75rem 1rem;
    }
    
    /* Chat message styling */
    .css-1eqt8fj {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        background-color: #f1f5f9;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #2563eb;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    .css-1d391kg > div {
        padding: 2rem 1rem;
    }
    
    /* Markdown styling */
    .markdown-text-container {
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Info message styling */
    .stAlert {
        border-radius: 6px;
        padding: 0.75rem 1rem;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background-color: #10b981;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background-color: #059669;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
        transform: translateY(-2px);
    }
    
    /* Card-like styling for components */
    .card-container {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure logging
set_log_level("INFO")  # Can be changed to DEBUG for more detailed logs

# --- Initialize session state ---
if "api_key" not in st.session_state:
    st.session_state.api_key = GEMINI_API_KEY or ""

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = """
    You are a research assistant that helps users by finding and synthesizing information from the web.
    You have access to search tools to gather information on different topics.
    Use these tools to provide comprehensive, accurate and well-sourced information.
    When using multiple tools, make sure to integrate the information logically.
    Always cite your sources when providing information.
    """

if "agent" not in st.session_state or st.session_state.reinitialize_agent:
    try:
        if st.session_state.get("api_key"):
            st.session_state.agent = ResearchAgent(
                api_key=st.session_state.api_key,
                model_name="gemini-2.0-flash",
                report_model_name="gemini-2.0-flash",
                output_dir="research_outputs",
                system_instruction=st.session_state.system_prompt
            )
            st.session_state.reinitialize_agent = False
            logger.info("Research Agent initialized successfully")
        else:
            st.session_state.agent = None
    except Exception as e:
        st.error(f"Failed to initialize Research Agent: {e}")
        logger.error(f"Failed to initialize Research Agent: {e}", exc_info=True)
        st.session_state.agent = None

if "research_results" not in st.session_state:
    st.session_state.research_results = None

if "research_status" not in st.session_state:
    st.session_state.research_status = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- App Header ---
st.markdown("""
<div style="text-align: center; padding: 1.5rem 0; background: linear-gradient(to right, #f0f9ff, #e0f2fe, #f0f9ff); border-radius: 10px; margin-bottom: 2rem;">
    <h1>🔍 AI Research Assistant</h1>
    <p style="font-size: 1.1rem; color: #334155; max-width: 700px; margin: 0 auto;">
        Your intelligent companion for comprehensive research, powered by Google Gemini
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card-container">
    <p style="font-size: 1.05rem; color: #334155; line-height: 1.6;">
        This tool helps you research topics by:
    </p>
    <ol style="font-size: 1.05rem; color: #334155; padding-left: 1.5rem; line-height: 1.6;">
        <li>Generating relevant search queries</li>
        <li>Searching the web for information</li>
        <li>Extracting content from web pages</li>
        <li>Creating a comprehensive research report</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# --- Sidebar for Settings ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <h2 style="margin-bottom: 0.5rem; font-size: 1.6rem;">Settings</h2>
        <div style="width: 50px; height: 3px; background: linear-gradient(90deg, #1e3a8a, #3b82f6); margin: 0 auto;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key input with better styling
    st.markdown('<p style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem; color: #334155;">Google Gemini API Key</p>', unsafe_allow_html=True)
    api_key = st.text_input("", value=st.session_state.api_key, type="password", placeholder="Enter your API key here", label_visibility="collapsed")
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        st.session_state.reinitialize_agent = True
        st.rerun()
    
    # System prompt customization
    st.markdown('<p style="font-size: 1.1rem; font-weight: 500; margin: 1rem 0 0.5rem 0; color: #334155;">System Prompt</p>', unsafe_allow_html=True)
    system_prompt = st.text_area(
        "",
        value=st.session_state.system_prompt,
        height=150,
        placeholder="Customize the agent's behavior...",
        label_visibility="collapsed"
    )
    if system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = system_prompt
        st.session_state.reinitialize_agent = True
    
    # Apply settings button
    if st.button("Apply Settings", key="apply_settings"):
        st.session_state.reinitialize_agent = True
        st.success("✅ Settings applied successfully!")
        st.rerun()
    
    st.markdown('<div style="margin: 1.5rem 0; border-top: 1px solid #e2e8f0;"></div>', unsafe_allow_html=True)
    
    # Log level selector with better styling
    st.markdown('<p style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem; color: #334155;">Log Level</p>', unsafe_allow_html=True)
    log_level = st.selectbox(
        "",
        options=["INFO", "DEBUG", "WARNING", "ERROR"],
        index=0,
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Apply Log Settings", key="apply_log"):
            set_log_level(log_level)
            st.success(f"✅ Log level set to {log_level}")
    
    st.markdown('<div style="margin: 1.5rem 0; border-top: 1px solid #e2e8f0;"></div>', unsafe_allow_html=True)
    
    # Stats and status
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <p style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem; color: #334155;">Status</p>
        <div style="width: 40px; height: 2px; background: linear-gradient(90deg, #1e3a8a, #3b82f6); margin: 0 auto 1rem auto;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.agent:
        st.markdown("""
        <div style="background-color: #ecfdf5; border-left: 4px solid #10b981; padding: 0.75rem; border-radius: 4px;">
            <p style="margin: 0; color: #065f46; font-weight: 500;">✅ Agent ready</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 0.75rem; border-radius: 4px;">
            <p style="margin: 0; color: #991b1b; font-weight: 500;">❌ Agent not initialized</p>
            <p style="margin: 0; font-size: 0.9rem; color: #991b1b;">Please enter API key</p>
        </div>
        """, unsafe_allow_html=True)
    
    # About section with better styling
    st.markdown('<div style="margin: 1.5rem 0; border-top: 1px solid #e2e8f0;"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <p style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem; color: #334155;">About</p>
        <div style="width: 40px; height: 2px; background: linear-gradient(90deg, #1e3a8a, #3b82f6); margin: 0 auto 1rem auto;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f8fafc; padding: 1rem; border-radius: 8px; font-size: 0.9rem; color: #334155; line-height: 1.5;">
        <p>This application uses the Gemini API from Google to perform research tasks. It searches the web, extracts information, and generates comprehensive reports on any topic.</p>
        <p style="margin-bottom: 0; font-style: italic;">Created with Streamlit, LangGraph, and Gemini.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Main Content Area ---
query_tab, chat_tab, history_tab = st.tabs(["Research", "Chat", "History"])

# --- Research Tab ---
with query_tab:
    st.markdown('<h2 style="margin-top: 0;">Research a Topic</h2>', unsafe_allow_html=True)
    
    # Research form with better styling
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    with st.form("research_form"):
        st.markdown('<p style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem; color: #334155;">What would you like to research?</p>', unsafe_allow_html=True)
        research_query = st.text_area(
            "",
            placeholder="Enter a topic or question to research...",
            height=100,
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("🔍 Start Research")
        with col2:
            clear_button = st.form_submit_button("🧹 Clear Results")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process form submission
    if submit_button and research_query:
        if not st.session_state.agent:
            st.markdown("""
            <div style="background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 1rem; border-radius: 6px; margin: 1rem 0;">
                <p style="margin: 0; color: #991b1b; font-weight: 500;">Please enter a valid Google Gemini API key in the sidebar to continue.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            try:
                # Create progress indicators with better styling
                progress_container = st.container()
                with progress_container:
                    st.markdown('<div class="card-container">', unsafe_allow_html=True)
                    status = st.empty()
                    progress_bar = st.progress(0)
                    details = st.empty()
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    def update_progress(message, progress_value):
                        status.info(message)
                        progress_bar.progress(progress_value)
                        logger.info(message)
                    
                    # Start research process
                    update_progress("Starting research process...", 0.05)
                    
                    # Step 1: Generate search queries
                    update_progress("Generating search queries...", 0.1)
                    time.sleep(0.5)  # Small delay for UI updates
                    
                    # Step 2: Run the agent
                    update_progress("Performing research. This may take a few minutes...", 0.2)
                    
                    # Run the agent asynchronously
                    try:
                        st.session_state.research_status = "in_progress"
                        results = st.session_state.agent.research(research_query)
                        st.session_state.research_results = results
                        st.session_state.research_status = "complete"
                        update_progress("Research completed!", 1.0)
                        time.sleep(1)  # Small delay for user to see completion
                        st.rerun()
                    except Exception as e:
                        st.session_state.research_status = "error"
                        logger.error(f"Research error: {e}", exc_info=True)
                        status.error(f"Error during research: {str(e)}")
                        details.error("Check the logs for more details.")
                        st.stop()
            
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                logger.error(f"Unexpected error: {e}", exc_info=True)
    
    if clear_button:
        st.session_state.research_results = None
        st.session_state.research_status = None
        st.rerun()
    
    # Display results with better styling
    if st.session_state.research_status == "complete" and st.session_state.research_results:
        results = st.session_state.research_results
        
        # Debug info
        with st.expander("Debug Info", expanded=False):
            st.write("Results type:", type(results))
            st.write("Results keys (if dict):", results.keys() if hasattr(results, "keys") else "Not a dict")
            st.write("Results attributes:", dir(results))
        
        st.markdown("""
        <div style="background-color: #ecfdf5; border-left: 4px solid #10b981; padding: 1rem; border-radius: 6px; margin: 1rem 0;">
            <p style="margin: 0; color: #065f46; font-weight: 500;">✅ Research completed successfully!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different parts of the results
        report_tab, data_tab = st.tabs(["📄 Report", "🔍 Research Data"])
        
        with report_tab:
            st.markdown('<h2 style="margin-top: 0.5rem;">Research Report</h2>', unsafe_allow_html=True)
            
            # Access report safely
            try:
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                if hasattr(results, "report"):
                    st.markdown(results.report)
                elif isinstance(results, dict) and "report" in results:
                    st.markdown(results["report"])
                else:
                    st.warning("Report not found in results.")
                    st.write("Available data:", results)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error displaying report: {e}")
                logger.error(f"Error displaying report: {e}", exc_info=True)
            
            # Download button for the report
            try:
                report_path = results.report_path if hasattr(results, "report_path") else (results.get("report_path") if isinstance(results, dict) else None)
                if report_path and os.path.exists(report_path):
                    with open(report_path, "r", encoding="utf-8") as f:
                        report_content = f.read()
                    st.download_button(
                        label="📥 Download Report (TXT)",
                        data=report_content,
                        file_name=os.path.basename(report_path),
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"Error with download button: {e}")
                logger.error(f"Error with download button: {e}", exc_info=True)
        
        with data_tab:
            st.markdown('<h2 style="margin-top: 0.5rem;">Research Data</h2>', unsafe_allow_html=True)
            
            # Display queries safely
            st.markdown('<h3>Generated Search Queries</h3>', unsafe_allow_html=True)
            try:
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                queries = results.queries if hasattr(results, "queries") else (results.get("queries", []) if isinstance(results, dict) else [])
                if queries:
                    for i, query in enumerate(queries, 1):
                        st.markdown(f'<div style="padding: 0.5rem; border-left: 3px solid #3b82f6; margin-bottom: 0.5rem; background-color: #f8fafc;"><p style="margin: 0; font-size: 1rem;"><strong>{i}.</strong> {query}</p></div>', unsafe_allow_html=True)
                else:
                    st.info("No queries available.")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error displaying queries: {e}")
                logger.error(f"Error displaying queries: {e}", exc_info=True)
            
            # Display search results safely
            st.markdown('<h3>Search Results</h3>', unsafe_allow_html=True)
            try:
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                search_results = results.search_results if hasattr(results, "search_results") else (results.get("search_results", {}) if isinstance(results, dict) else {})
                if search_results:
                    for query, links in search_results.items():
                        st.markdown(f'<div style="padding: 0.75rem; border-radius: 6px; margin-bottom: 1rem; background-color: #f8fafc;"><p style="margin: 0 0 0.5rem 0; font-weight: 500; color: #334155;">Query: {query}</p>', unsafe_allow_html=True)
                        for link in links:
                            st.markdown(f'<p style="margin: 0.25rem 0; font-size: 0.95rem;">• <a href="{link}" target="_blank">{link}</a></p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No search results available.")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error displaying search results: {e}")
                logger.error(f"Error displaying search results: {e}", exc_info=True)
            
            # Display extracted content safely
            st.markdown('<h3>Extracted Content</h3>', unsafe_allow_html=True)
            try:
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                extracted_contents = results.extracted_contents if hasattr(results, "extracted_contents") else (results.get("extracted_contents", []) if isinstance(results, dict) else [])
                if extracted_contents:
                    if st.checkbox("Show extracted content (may be lengthy)"):
                        for item in extracted_contents:
                            st.markdown(f'<div style="padding: 0.75rem; border-radius: 6px; margin-bottom: 1rem; background-color: #f8fafc; border-left: 3px solid #3b82f6;">', unsafe_allow_html=True)
                            st.markdown(f'<p style="margin: 0 0 0.5rem 0; font-weight: 500; color: #334155;">Source: <a href="{item["url"]}" target="_blank">{item["url"]}</a></p>', unsafe_allow_html=True)
                            st.markdown(f'<p style="margin: 0 0 0.5rem 0; font-size: 0.95rem;">Query: {item["query"]}</p>', unsafe_allow_html=True)
                            with st.expander("View Content"):
                                st.text(item['content'])
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No extracted content available.")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error displaying extracted content: {e}")
                logger.error(f"Error displaying extracted content: {e}", exc_info=True)
            
            # Download JSON data safely
            try:
                json_path = results.json_path if hasattr(results, "json_path") else (results.get("json_path") if isinstance(results, dict) else None)
                if json_path and os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        json_content = f.read()
                    st.download_button(
                        label="📥 Download Research Data (JSON)",
                        data=json_content,
                        file_name=os.path.basename(json_path),
                        mime="application/json"
                    )
            except Exception as e:
                st.error(f"Error with JSON download: {e}")
                logger.error(f"Error with JSON download: {e}", exc_info=True)

# --- Chat Tab ---
with chat_tab:
    st.markdown('<h2 style="margin-top: 0.5rem;">Chat with Research Assistant</h2>', unsafe_allow_html=True)
    
    if not st.session_state.agent:
        st.markdown("""
        <div style="background-color: #fff7ed; border-left: 4px solid #f97316; padding: 1rem; border-radius: 6px; margin: 1rem 0;">
            <p style="margin: 0; color: #9a3412; font-weight: 500;">Please enter a valid Google Gemini API key in the sidebar to use the chat feature.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display chat history with better styling
        st.markdown('<div class="card-container" style="padding: 0; overflow: hidden;">', unsafe_allow_html=True)
        
        # Container for chat messages with scrolling
        chat_container = st.container()
        with chat_container:
            for i, (role, message) in enumerate(st.session_state.chat_history):
                if role == "user":
                    st.markdown(f"""
                    <div style="display: flex; margin-bottom: 1rem;">
                        <div style="width: 40px; height: 40px; border-radius: 50%; background-color: #e0f2fe; display: flex; align-items: center; justify-content: center; margin-right: 1rem;">
                            <span style="font-weight: bold; color: #0284c7;">You</span>
                        </div>
                        <div style="background-color: #e0f2fe; padding: 1rem; border-radius: 8px; flex-grow: 1; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                            <p style="margin: 0; color: #0c4a6e;">{message}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; margin-bottom: 1rem;">
                        <div style="width: 40px; height: 40px; border-radius: 50%; background-color: #f0f9ff; display: flex; align-items: center; justify-content: center; margin-right: 1rem;">
                            <span style="font-weight: bold; color: #0369a1;">AI</span>
                        </div>
                        <div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; flex-grow: 1; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                            <p style="margin: 0; color: #0c4a6e;">{message}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input with better styling
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        with st.form("chat_form"):
            st.markdown('<p style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem; color: #334155;">Message the assistant</p>', unsafe_allow_html=True)
            user_message = st.text_area(
                "",
                placeholder="Ask a question or request information...",
                height=100,
                label_visibility="collapsed"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                chat_submit = st.form_submit_button("📤 Send Message")
            with col2:
                clear_chat = st.form_submit_button("🧹 Clear Chat")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process chat inputs
        if chat_submit and user_message:
            # Add user message to history
            st.session_state.chat_history.append(("user", user_message))
            
            # Get response from agent
            try:
                with st.spinner("Thinking..."):
                    response = st.session_state.agent.chat(user_message)
                
                # Add assistant response to history
                st.session_state.chat_history.append(("assistant", response))
                st.rerun()
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")
                logger.error(f"Chat error: {e}", exc_info=True)
        
        if clear_chat:
            st.session_state.chat_history = []
            st.rerun()

# --- History Tab ---
with history_tab:
    st.markdown('<h2 style="margin-top: 0.5rem;">Research History</h2>', unsafe_allow_html=True)
    
    # Scan research_outputs directory
    try:
        output_dir = "research_outputs"
        os.makedirs(output_dir, exist_ok=True)
        files = os.listdir(output_dir)
        
        # Group files by research session
        sessions = {}
        for file in files:
            if file.endswith(".txt") or file.endswith(".json"):
                # Extract base name (without _report.txt or .json)
                base_name = file.split("_report.txt")[0].split(".json")[0]
                if base_name not in sessions:
                    sessions[base_name] = {"reports": [], "data": []}
                
                if file.endswith("_report.txt"):
                    sessions[base_name]["reports"].append(file)
                else:
                    sessions[base_name]["data"].append(file)
        
        if not sessions:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background-color: #f8fafc; border-radius: 8px; margin: 1rem 0;">
                <p style="font-size: 1.1rem; color: #64748b;">No research history found.</p>
                <p style="font-size: 0.9rem; color: #64748b;">Your completed research sessions will appear here.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Create a grid layout for research history cards
            col1, col2 = st.columns(2)
            col_index = 0
            
            for base_name, files in sessions.items():
                # Alternate between columns
                current_col = col1 if col_index % 2 == 0 else col2
                col_index += 1
                
                with current_col:
                    st.markdown(f"""
                    <div style="background-color: #ffffff; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); 
                         margin-bottom: 1rem; overflow: hidden;">
                        <div style="background: linear-gradient(to right, #2563eb, #3b82f6); padding: 1rem; color: white;">
                            <h3 style="margin: 0; font-size: 1.2rem;">{base_name.replace('_', ' ')}</h3>
                        </div>
                        <div style="padding: 1rem;">
                    """, unsafe_allow_html=True)
                    
                    # Display available files
                    for report in files["reports"]:
                        file_path = os.path.join(output_dir, report)
                        with open(file_path, "r", encoding="utf-8") as f:
                            report_content = f.read()
                        
                        st.download_button(
                            label=f"📄 TXT Report",
                            data=report_content,
                            file_name=report,
                            mime="text/plain",
                            key=f"download_txt_{report}"
                        )
                    
                    for data_file in files["data"]:
                        file_path = os.path.join(output_dir, data_file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            file_content = f.read()
                        st.download_button(
                            label=f"📊 Raw Data (JSON)",
                            data=file_content,
                            file_name=data_file,
                            mime="application/json",
                            key=f"download_{data_file}"
                        )
                    
                    st.markdown("</div></div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading research history: {str(e)}")
        logger.error(f"History error: {e}", exc_info=True)

# Footer with better styling
st.markdown("---")
st.markdown("""
<footer style="text-align: center; padding: 1.5rem; background: linear-gradient(to right, #f0f9ff, #e0f2fe, #f0f9ff); border-radius: 10px; margin-top: 2rem;">
    <p style="font-size: 0.9rem; color: #334155; margin: 0;">
        Built with ❤️ using Streamlit, LangGraph, and Gemini API.
    </p>
    <p style="font-size: 0.8rem; color: #64748b; margin: 0.5rem 0 0 0;">
        This tool performs web searches and content extraction for research purposes.
    </p>
</footer>
""", unsafe_allow_html=True)