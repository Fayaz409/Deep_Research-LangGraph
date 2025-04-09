"""
This module contains all the tools used by the research agent.
Each tool is defined as a function that can be used by the agent.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
import re
from duckduckgo_search import DDGS
import logging
from urllib.parse import urlparse
import time

# Constants for the tools
MAX_LINKS_PER_SEARCH = 5
MAX_CONTENT_LENGTH = 4000
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
}

# Response models for each tool
class SearchResult(BaseModel):
    links: List[str] = Field(description="List of relevant web links")
    
class ContentResult(BaseModel):
    content: str = Field(description="Extracted content from the webpage")
    success: bool = Field(description="Whether content extraction was successful")
    url: str = Field(description="The URL that was scraped")

class GeneratedQueriesResult(BaseModel):
    queries: List[str] = Field(description="List of generated search queries")

# Tool functions
def search_duck_duck_go(query: str, max_results: int = MAX_LINKS_PER_SEARCH) -> SearchResult:
    """
    Search DuckDuckGo for relevant web pages.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        A SearchResult containing links
    """
    links = []
    try:
        with DDGS(headers=HEADERS, timeout=20) as ddgs:
            results = ddgs.text(query, region='wt-wt', safesearch='moderate', max_results=max_results + 5)
            count = 0
            
            if results:
                for result in results:
                    if 'href' in result:
                        href_lower = result['href'].lower()
                        # Filter out file types we don't want
                        if not any(ext in href_lower for ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.zip', '.rar', '.jpg', '.png', '.gif', '.svg']):
                            # Check if the URL is valid
                            try:
                                parsed = urlparse(result['href'])
                                if parsed.scheme and parsed.netloc:
                                    links.append(result['href'])
                                    count += 1
                            except Exception:
                                continue
                            
                    if count >= max_results:
                        break
                        
    except Exception as e:
        logging.error(f"Error during DuckDuckGo search: {e}")
    
    return SearchResult(links=links)

def get_page_content(url: str, max_length: int = MAX_CONTENT_LENGTH) -> ContentResult:
    """
    Fetch and extract content from a webpage.
    
    Args:
        url: The URL to fetch content from
        max_length: Maximum length of content to return
        
    Returns:
        A ContentResult containing the extracted content
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            return ContentResult(
                content=f"Cannot extract content: not HTML (content-type: {content_type})",
                success=False,
                url=url
            )
        
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "noscript"]):
            element.decompose()
        
        # Try to find main content first
        main_content_tags = ['article', 'main', '[role="main"]']
        main_content = None
        for tag in main_content_tags:
            try:
                main_content = soup.select_one(tag)
                if main_content:
                    break
            except Exception:
                continue
        
        text_parts = []
        if main_content:
            tags_to_extract = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'td', 'th'], recursive=True)
            for tag in tags_to_extract:
                text = tag.get_text(separator=' ', strip=True)
                if len(text) > 25 and not text.lower().startswith(('copyright', 'related posts', 'leave a reply')):
                    text_parts.append(text)
        else:
            # Fallback to all paragraphs
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text(separator=' ', strip=True)
                if len(text) > 25:
                    text_parts.append(text)
        
        if not text_parts:
            return ContentResult(
                content=f"No significant content found on the page.",
                success=False,
                url=url
            )
        
        full_text = ' '.join(text_parts)
        full_text = re.sub(r'\s{2,}', ' ', full_text).strip()
        
        # Add URL as source at the beginning
        content_with_source = f"Content from: {url}\n\n{full_text[:max_length]}"
        if len(full_text) > max_length:
            content_with_source += "..."
        
        return ContentResult(
            content=content_with_source,
            success=True,
            url=url
        )
        
    except requests.exceptions.Timeout:
        return ContentResult(
            content=f"Timeout error when fetching the page.",
            success=False,
            url=url
        )
    except requests.exceptions.RequestException as e:
        return ContentResult(
            content=f"Error fetching the page: {str(e)}",
            success=False,
            url=url
        )
    except Exception as e:
        return ContentResult(
            content=f"Unexpected error processing the page: {str(e)}",
            success=False,
            url=url
        )

def generate_search_queries(query: str, model, num_queries: int = 5) -> GeneratedQueriesResult:
    """
    Generate related search queries using an LLM.
    
    Args:
        query: The original user query
        model: The LLM model to use for generation
        num_queries: Number of queries to generate
        
    Returns:
        A GeneratedQueriesResult containing the list of generated queries
    """
    prompt = f"""
    Given the user's query: "{query}"
    Generate a list of {num_queries} specific and diverse search engine queries that, when researched individually, 
    would help provide a comprehensive and well-structured explanation of the original query. 
    Focus on different facets like definitions, core concepts, examples, benefits, drawbacks, applications, or related topics.
    
    Output ONLY a valid JSON list of strings. Do not include any other text.
    Example format:
    ["query about definition", "query about applications", "query about examples", "query about benefits", "query about related concepts"]
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response to extract JSON
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Try to parse as JSON
        import json
        queries_list = json.loads(response_text)
        
        # Validate and filter
        if isinstance(queries_list, list) and all(isinstance(item, str) for item in queries_list):
            return GeneratedQueriesResult(queries=queries_list[:num_queries])
        else:
            # Fallback if JSON structure is not as expected
            logging.warning("LLM response was not a valid list of strings")
            return GeneratedQueriesResult(queries=[query])  # Return original query as fallback
    
    except Exception as e:
        logging.error(f"Error generating search queries: {e}")
        return GeneratedQueriesResult(queries=[query])  # Return original query as fallback

def search_wikipedia(query: str, max_results: int = 3) -> SearchResult:
    """
    Search Wikipedia for relevant articles.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        A SearchResult containing links to Wikipedia articles
    """
    try:
        import wikipedia
        search_results = wikipedia.search(query, results=max_results)
        links = [f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}" for title in search_results]
        return SearchResult(links=links)
    except Exception as e:
        logging.error(f"Error during Wikipedia search: {e}")
        return SearchResult(links=[])

def get_wikipedia_page(title: str) -> ContentResult:
    """
    Get content from a Wikipedia page.
    
    Args:
        title: The title of the Wikipedia page
        
    Returns:
        A ContentResult containing the extracted content
    """
    try:
        import wikipedia
        page = wikipedia.page(title)
        content = page.content[:MAX_CONTENT_LENGTH]
        url = page.url
        return ContentResult(
            content=f"Content from Wikipedia: {url}\n\n{content}{'...' if len(page.content) > MAX_CONTENT_LENGTH else ''}",
            success=True,
            url=url
        )
    except Exception as e:
        return ContentResult(
            content=f"Error getting Wikipedia page: {str(e)}",
            success=False,
            url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        )