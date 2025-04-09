"""
This module implements the ResearchAgent using langgraph.
"""

import os
import json
import time
import datetime
from typing import List, Dict, Any, Callable, Annotated
from operator import add
from pydantic import BaseModel, Field

from langgraph.graph import END, START, StateGraph
from google.api_core import retry
from google.generativeai.types import RequestOptions
import google.generativeai as genai

from logger import logger
from tools import (
    search_duck_duck_go,
    get_page_content,
    generate_search_queries,
    search_wikipedia,
    get_wikipedia_page
)

# Define the agent state
class AgentState(BaseModel):
    """
    State tracked by the agent during execution.
    
    Attributes:
        messages: The messages exchanged in the conversation.
        queries: Generated search queries from the original query.
        search_results: Results from searches performed.
        extracted_contents: Content extracted from web pages.
        report: The final report generated.
        user_query: The original user query.
    """
    messages: Annotated[list, add] = Field(default_factory=list)
    queries: List[str] = Field(default_factory=list)
    search_results: Dict[str, List[str]] = Field(default_factory=dict)
    extracted_contents: List[Dict[str, Any]] = Field(default_factory=list)
    report: str = ""
    user_query: str = ""
    json_path: str = ""
    report_path: str = ""


class ResearchAgent:
    """
    An agent that performs web research and generates reports based on user queries.
    """
    
    def __init__(
        self, 
        api_key: str = None,
        model_name: str = "gemini-2.0-flash",
        report_model_name: str = "gemini-2.0-flash",
        output_dir: str = "research_outputs",
        system_instruction: str = None
    ):
        """
        Initialize the research agent.
        
        Args:
            api_key: The Gemini API key.
            model_name: The name of the model to use for the agent.
            report_model_name: The name of the model to use for report generation.
            output_dir: Directory to save outputs.
            system_instruction: Custom system instructions for the model.
        """
        # Setup API key
        if api_key:
            self.api_key = api_key
            genai.configure(api_key=api_key)
        elif os.getenv("GEMINI_API_KEY"):
            self.api_key = os.getenv("GEMINI_API_KEY")
            genai.configure(api_key=self.api_key)
        else:
            raise ValueError("No API key provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        # Setup default system instruction if none provided
        if system_instruction is None:
            system_instruction = """
            You are a research assistant that helps users by finding and synthesizing information from the web.
            You have access to search tools to gather information on different topics.
            Use these tools to provide comprehensive, accurate and well-sourced information.
            When using multiple tools, make sure to integrate the information logically.
            Always cite your sources when providing information.
            """
        
        self.system_instruction = system_instruction
        
        # Setup models
        self.model_name = model_name
        self.report_model_name = report_model_name
        
        # Create model with tools and system instruction
        self.model = genai.GenerativeModel(
            self.model_name,
            tools=[
                search_duck_duck_go,
                get_page_content,
                search_wikipedia,
                get_wikipedia_page
            ],
            system_instruction=self.system_instruction
        )
        
        self.report_model = genai.GenerativeModel(
            self.report_model_name,
            system_instruction=self.system_instruction
        )
        
        # Setup tools
        self.tools = [
            search_duck_duck_go,
            get_page_content,
            search_wikipedia,
            get_wikipedia_page
        ]
        self.tool_mapping = {tool.__name__: tool for tool in self.tools}
        
        # Setup output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Build the agent graph
        self.graph = None
        self.build_agent()
        
        logger.info(f"Research Agent initialized with models: {model_name} and {report_model_name}")
    
    def generate_queries(self, state: AgentState) -> AgentState:
        """
        Generate search queries based on the user query.
        
        Args:
            state: The current agent state.
            
        Returns:
            Updated agent state with generated queries.
        """
        logger.info(f"Generating search queries for: {state.user_query}")
        
        # Use the model to generate search queries
        result = generate_search_queries(state.user_query, self.model)
        
        # Update the state
        state.queries = result.queries
        logger.info(f"Generated {len(state.queries)} search queries")
        
        return state
    
    def perform_searches(self, state: AgentState) -> AgentState:
        """
        Perform web searches for each generated query.
        
        Args:
            state: The current agent state.
            
        Returns:
            Updated agent state with search results.
        """
        logger.info("Performing web searches...")
        
        for query in state.queries:
            logger.info(f"Searching for: {query}")
            
            # Search DuckDuckGo
            ddg_results = search_duck_duck_go(query)
            
            # Search Wikipedia
            wiki_results = search_wikipedia(query)
            
            # Combine results
            all_links = ddg_results.links + wiki_results.links
            
            # Remove duplicates while preserving order
            unique_links = []
            for link in all_links:
                if link not in unique_links:
                    unique_links.append(link)
            
            # Update state
            state.search_results[query] = unique_links
            logger.info(f"Found {len(unique_links)} links for query: {query}")
            
            # Add a small delay between searches
            time.sleep(2)
        
        return state
    
    def extract_content(self, state: AgentState) -> AgentState:
        """
        Extract content from each search result.
        
        Args:
            state: The current agent state.
            
        Returns:
            Updated agent state with extracted content.
        """
        logger.info("Extracting content from web pages...")
        
        for query, links in state.search_results.items():
            for link in links:
                logger.info(f"Extracting content from: {link}")
                
                # Handle Wikipedia links specially
                if "wikipedia.org/wiki/" in link:
                    title = link.split("/wiki/")[-1].replace("_", " ")
                    content_result = get_wikipedia_page(title)
                else:
                    content_result = get_page_content(link)
                
                if content_result.success:
                    # Add to extracted contents
                    state.extracted_contents.append({
                        "query": query,
                        "url": link,
                        "content": content_result.content
                    })
                    logger.info(f"Successfully extracted content from {link}")
                else:
                    logger.warning(f"Failed to extract content from {link}: {content_result.content}")
                
                # Add a small delay between requests
                time.sleep(3)
        
        return state
    
    def generate_report(self, state: AgentState) -> AgentState:
        """
        Generate a report based on extracted content.
        
        Args:
            state: The current agent state.
            
        Returns:
            Updated agent state with the generated report.
        """
        logger.info("Generating final report...")
        
        # Prepare context from extracted content
        context_parts = []
        
        for item in state.extracted_contents:
            context_parts.append(f"Source: {item['url']}\nGenerated Query: {item['query']}\nContent:\n{item['content']}\n\n---\n")
        
        if not context_parts:
            logger.warning("No content was extracted. Cannot generate report.")
            state.report = "Error: No content was successfully extracted from web searches to generate a report."
            return state
        
        # Join context parts
        full_context = "\n".join(context_parts)
        
        # Generate report using the model
        prompt = f"""
        **Original User Query:**
        {state.user_query}

        **Context Gathered from Web Search:**
        --- START CONTEXT ---
        {full_context}
        --- END CONTEXT ---

        **Task:**
        Craft an exceptionally detailed, comprehensive, and engaging report that directly addresses the original user query. Your primary goal is to transform the provided context into a rich narrative, explaining each concept with significant depth and clarity, making the information not just informative but also uniquely interesting to read. Your response MUST be based exclusively on the provided "Context Gathered from Web Search" above.

        **Instructions:**

        1.  **Go Beyond Surface Level:** Do not just summarize. For *every* topic and sub-topic derived from the context, elaborate **extensively** and **verbosely**. Dive deep into the nuances, explaining the 'why' and 'how' behind each piece of information presented in the context.
        2.  **Rich and Engaging Explanations:** Use descriptive language and provide illustrative details drawn *only* from the context to make the explanations vivid and captivating. Explain the significance and implications of each feature or concept discussed. If the context mentions a benefit, elaborate profoundly on *how* it benefits the user/developer and *why* it's important.
        3.  **Maximum Detail is Key:** Extract and present every relevant detail from the context. Assume the reader wants the most thorough understanding possible based *solely* on the provided text. Expand on definitions, functionalities, and comparisons significantly.
        4.  **Strict Context Adherence:** Base your *entire* report strictly and exclusively on the information found within the provided "Context Gathered from Web Search". Do not introduce external knowledge or examples.
        5.  **Logical Structure and Clarity:** Organize the report logically using clear sections and headings (markdown format: # for main headings, ## for subheadings, ### for sub-subheadings). Ensure smooth transitions between topics.
        6.  **Source Citation:** Meticulously cite the source URL immediately after presenting any piece of information, referencing the corresponding URL from the context.
        7.  **Acknowledge Limitations:** If the provided context lacks sufficient detail to elaborate extensively on a specific aspect relevant to the user query, explicitly state this limitation.
        8.  **Comprehensive Coverage:** Ensure all aspects of the original user query that are addressed in the context are covered in exhaustive detail in the final report.
        9.  **Markdown Formatting:** Use markdown for structure (#, ##, ### headings) and potentially for emphasis (like bolding key terms *if justified by the context's emphasis*), but prioritize clear prose.

        Generate the deeply detailed and engaging report now.
        """
        
        try:
            response = self.report_model.generate_content(prompt)
            state.report = response.text
            logger.info("Report generated successfully")
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            state.report = f"Error generating report: {str(e)}"
        
        return state
    
    def save_outputs(self, state: AgentState) -> AgentState:
        """
        Save the research data and report to files.
        
        Args:
            state: The current agent state.
            
        Returns:
            Updated agent state with file paths.
        """
        logger.info("Saving outputs to files...")
        
        # Create timestamp and sanitized query name for filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = state.user_query.lower().replace(" ", "_")
        safe_query = "".join(c for c in safe_query if c.isalnum() or c == "_")
        safe_query = safe_query[:50]  # Limit length
        
        # Prepare data for JSON
        json_data = {
            "initial_query": state.user_query,
            "generated_queries": state.queries,
            "search_results": state.search_results,
            "extracted_contents": state.extracted_contents
        }
        
        # Save JSON data
        json_filename = f"{safe_query}_{timestamp}.json"
        json_path = os.path.join(self.output_dir, json_filename)
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            state.json_path = json_path
            logger.info(f"Data saved to {json_path}")
        except Exception as e:
            logger.error(f"Error saving JSON data: {e}")
        
        # Save report
        report_filename = f"{safe_query}_{timestamp}_report.txt"
        report_path = os.path.join(self.output_dir, report_filename)
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"Original Query: {state.user_query}\n\n")
                f.write("="*20 + " GENERATED REPORT " + "="*20 + "\n\n")
                f.write(state.report)
            state.report_path = report_path
            logger.info(f"Report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
        
        return state
    
    def call_llm(self, state: AgentState) -> dict:
        """
        Call the LLM with the current messages.
        
        Args:
            state: The current agent state.
            
        Returns:
            Updated messages.
        """
        try:
            response = self.model.generate_content(
                state.messages,
                request_options=RequestOptions(
                    retry=retry.Retry(initial=10, multiplier=2, maximum=60, timeout=300)
                ),
            )
            return {
                "messages": [
                    type(response.candidates[0].content).to_dict(
                        response.candidates[0].content
                    )
                ]
            }
        except Exception as e:
            logger.error(f"Error calling LLM: {e}", exc_info=True)
            # Return an error message if the call fails
            return {
                "messages": [
                    {"role": "model", "parts": [{"text": f"Error: {str(e)}"}]}
                ]
            }

    def use_tool(self, state: AgentState) -> dict:
        """
        Use a tool based on the function call from the LLM.
        
        Args:
            state: The current agent state.
            
        Returns:
            Tool response to add to messages.
        """
        assert any("function_call" in part for part in state.messages[-1]["parts"])
        tool_result_parts = []
        
        for part in state.messages[-1]["parts"]:
            if "function_call" in part:
                name = part["function_call"]["name"]
                func = self.tool_mapping[name]
                try:
                    result = func(**part["function_call"]["args"])
                    tool_result_parts.append(
                        {
                            "function_response": {
                                "name": name,
                                "response": result.model_dump(mode="json"),
                            }
                        }
                    )
                except Exception as e:
                    logger.error(f"Error using tool {name}: {e}", exc_info=True)
                    tool_result_parts.append(
                        {
                            "function_response": {
                                "name": name,
                                "response": {"success": False, "content": f"Error: {str(e)}"},
                            }
                        }
                    )
        
        return {"messages": [{"role": "tool", "parts": tool_result_parts}]}

    @staticmethod
    def should_we_stop(state: AgentState) -> str:
        """
        Decide if we should continue using tools or stop.
        
        Args:
            state: The current agent state.
            
        Returns:
            Next node to execute or END.
        """
        logger.debug(f"Checking if agent should stop")
        
        if state.messages and "parts" in state.messages[-1] and any("function_call" in part for part in state.messages[-1]["parts"]):
            logger.debug("Agent will use a tool")
            return "use_tool"
        else:
            logger.debug("Agent will stop")
            return END

    def build_agent(self):
        """
        Build the agent graph.
        """
        # Create main langgraph for the agent conversation
        conversation_graph = StateGraph(AgentState)
        conversation_graph.add_node("call_llm", self.call_llm)
        conversation_graph.add_node("use_tool", self.use_tool)
        conversation_graph.add_edge(START, "call_llm")
        conversation_graph.add_conditional_edges("call_llm", self.should_we_stop)
        conversation_graph.add_edge("use_tool", "call_llm")
        
        # Create the orchestration graph
        workflow = StateGraph(AgentState)
        workflow.add_node("generate_queries", self.generate_queries)
        workflow.add_node("perform_searches", self.perform_searches)
        workflow.add_node("extract_content", self.extract_content)
        workflow.add_node("generate_report", self.generate_report)
        workflow.add_node("save_outputs", self.save_outputs)
        
        # Connect the nodes
        workflow.add_edge(START, "generate_queries")
        workflow.add_edge("generate_queries", "perform_searches")
        workflow.add_edge("perform_searches", "extract_content")
        workflow.add_edge("extract_content", "generate_report")
        workflow.add_edge("generate_report", "save_outputs")
        workflow.add_edge("save_outputs", END)
        
        # Compile the graph
        self.workflow_graph = workflow.compile()
        self.conversation_graph = conversation_graph.compile()
    
    def research(self, query: str) -> AgentState:
        """
        Perform a complete research workflow.
        
        Args:
            query: The user query to research.
            
        Returns:
            The final agent state with all results.
        """
        logger.info(f"Starting research for query: {query}")
        
        # Initialize state
        initial_state = AgentState(user_query=query)
        
        # Run the workflow
        try:
            final_state = self.workflow_graph.invoke(initial_state)
            logger.info("Research workflow completed successfully")
            return final_state
        except Exception as e:
            logger.error(f"Error in research workflow: {e}", exc_info=True)
            raise
    
    def chat(self, query: str) -> str:
        """
        Have a conversation with the agent.
        
        Args:
            query: The user query.
            
        Returns:
            The agent's response.
        """
        initial_state = AgentState(
            messages=[{"role": "user", "parts": [{"text": query}]}],
        )
        
        try:
            output_state = self.conversation_graph.invoke(initial_state)
            
            # Extract the final response
            final_message = output_state["messages"][-1]
            if "parts" in final_message and len(final_message["parts"]) > 0:
                # Find text part
                for part in final_message["parts"]:
                    if "text" in part:
                        return part["text"]
                # Fallback to last part if no text part found
                return str(final_message["parts"][-1])
            else:
                return "No response generated."
        except Exception as e:
            logger.error(f"Error in agent conversation: {e}", exc_info=True)
            return f"Error: {str(e)}"
