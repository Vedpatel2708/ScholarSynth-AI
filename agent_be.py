from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_core.tools import FunctionTool
from groq_client import GroqChatCompletionClient
import arxiv
import json
from typing import List, Dict
import asyncio
import os

# Tool function to fetch papers from arXiv
def get_papers_sync(query: str, max_results: int = 5) -> str:
    """Fetch papers from arXiv and return as JSON string"""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    try:
        for result in client.results(search):
            papers.append({
                "title": result.title.strip(),
                "authors": [a.name for a in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
                "summary": result.summary.strip().replace('\n', ' ')[:400] + "...",
                "pdf_url": result.pdf_url
            })
    except Exception as e:
        print(f"Error fetching papers: {e}")
        return json.dumps([])
    return json.dumps(papers, indent=2)

# Helper function to create formatted literature review
def create_formatted_review(topic: str, papers_json: str) -> str:
    """Create a properly formatted literature review from papers JSON"""
    try:
        papers = json.loads(papers_json)
        
        if not papers:
            return f"# Literature Review: {topic}\n\n❌ No papers found for this topic."
        
        # Start building the review
        review = f"""# Literature Review: {topic}

## Introduction

This literature review examines recent research developments in the field of {topic}. The following analysis is based on {len(papers)} relevant papers retrieved from arXiv, providing insights into current methodologies, key contributions, and emerging trends in this domain.

## Paper Analysis

"""
        
        # Process each paper
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', 'Unknown Title')
            authors = paper.get('authors', [])
            published = paper.get('published', 'Unknown Date')
            summary = paper.get('summary', 'No summary available')
            pdf_url = paper.get('pdf_url', '#')
            
            # Format authors
            if len(authors) > 3:
                author_str = f"{', '.join(authors[:3])} et al."
            else:
                author_str = ', '.join(authors)
            
            # Clean and truncate summary
            clean_summary = summary.replace('...', '').strip()
            if len(clean_summary) > 300:
                clean_summary = clean_summary[:300] + "..."
            
            review += f"""### {i}. [{title}]({pdf_url})

**Authors:** {author_str}  
**Published:** {published}

**Summary:** {clean_summary}

**Key Contributions:** This work contributes to the {topic} field by addressing important research questions and presenting novel approaches to existing challenges.

---

"""
        
        # Add conclusion
        review += f"""## Conclusion

The reviewed papers demonstrate significant progress in {topic} research. Key themes across the literature include methodological innovations, practical applications, and theoretical advancements. Future research directions may focus on addressing current limitations and exploring new applications of these techniques.

## References

All papers are available through arXiv and can be accessed via the provided links above.

---
*Literature review generated using AutoGen multi-agent system*"""
        
        return review
        
    except Exception as e:
        return f"""# Literature Review: {topic}

## Error

❌ Error processing papers: {str(e)}

## Raw Data
{papers_json}
"""

# Simplified Literature Review System
class LitReviewSystem:
    def __init__(self):
        self.model_client = None
        self.researcher = None
        self.summarizer = None
        self.papers_data = None

    def _get_api_key(self):
        """Get API key from multiple sources"""
        api_key = None
        
        # Try environment variable first
        api_key = os.getenv("GROQ_API_KEY")
        if api_key and api_key.strip():
            return api_key.strip()
        
        # Try streamlit session state
        try:
            import streamlit as st
            if hasattr(st, 'session_state') and 'groq_api_key' in st.session_state:
                session_key = st.session_state.groq_api_key
                if session_key and session_key.strip():
                    # Set environment variable for consistency
                    os.environ["GROQ_API_KEY"] = session_key.strip()
                    return session_key.strip()
        except ImportError:
            pass
            
        return None

    def _initialize_clients(self):
        """Initialize clients only when needed"""
        if self.model_client is None:
            # Get API key
            api_key = self._get_api_key()
            
            if not api_key:
                raise ValueError("GROQ_API_KEY is not available. Please set it in environment variables or Streamlit session state.")
            
            # Initialize Groq Client with explicit API key
            self.model_client = GroqChatCompletionClient(api_key=api_key)
            
            # Create AutoGen FunctionTool
            get_papers_tool = FunctionTool(
                get_papers_sync,
                name="get_papers",
                description="Fetch academic papers from arXiv based on a search query"
            )
            
            # Researcher Agent with tool
            self.researcher = AssistantAgent(
                name="researcher",
                model_client=self.model_client,
                system_message=(
                    "You are a research assistant. When given a topic, use the get_papers tool to fetch relevant papers from arXiv. "
                    "After fetching the papers, return them as a properly formatted JSON array containing title, authors, "
                    "published date, summary, and pdf_url for each paper. "
                    "Always use the get_papers tool first, then format the response."
                ),
                tools=[get_papers_tool]
            )
            
            # Summarizer Agent
            self.summarizer = AssistantAgent(
                name="summarizer",
                model_client=self.model_client,
                system_message=(
                    "You are an expert academic researcher who creates literature reviews. "
                    "When given a JSON array of papers, create a comprehensive markdown literature review with:\n"
                    "1. A brief introduction (2-3 sentences) about the research topic\n"
                    "2. For each paper, create a section with:\n"
                    "   - Title as a markdown link to the PDF\n" 
                    "   - Authors and publication date\n"
                    "   - Key problem addressed\n"
                    "   - Main contributions and findings\n"
                    "3. A conclusion summarizing key themes and future research directions\n\n"
                    "Make it professional and academic in tone."
                )
            )

    async def run_stream(self, task: str):
        """Custom run method for literature review workflow"""
        topic = task.replace("Conduct a literature review on the topic: ", "")
        
        try:
            # Initialize clients if needed
            self._initialize_clients()
            
            # Step 1: Researcher fetches papers
            # Call the tool directly for more reliable results
            papers_json = get_papers_sync(topic, 5)
            
            if not papers_json or papers_json == "[]":
                yield TextMessage(content="❌ No papers found for this topic. Please try a different search term.", source="researcher")
                return
                
            self.papers_data = papers_json
            
            # Step 2: Create formatted review
            # Option 1: Try using the AI model client directly for more sophisticated review
            try:
                summarizer_prompt = f"""Create a comprehensive academic literature review for the topic '{topic}'. 

Here are the papers to analyze:
{papers_json}

Please create a professional literature review with:
1. An introduction explaining the importance of {topic}
2. For each paper: title (as link), authors, key problem, main contributions, and methodology
3. A conclusion highlighting key themes and future directions

Format it in clean markdown."""

                # Create messages in the format expected by Groq API
                formatted_messages = [{"role": "user", "content": summarizer_prompt}]
                response = await self.model_client.create(formatted_messages)
                ai_review = response.choices[0].message.content
                
                # Clean up the AI response
                if ai_review and len(ai_review.strip()) > 100:
                    yield TextMessage(content=ai_review, source="summarizer")
                else:
                    raise Exception("AI response too short or empty")
                    
            except Exception as model_error:
                print(f"AI model error: {model_error}")
                # Option 2: Use formatted template as fallback
                formatted_review = create_formatted_review(topic, papers_json)
                yield TextMessage(content=formatted_review, source="summarizer")
            
        except ValueError as ve:
            # Handle API key error specifically
            yield TextMessage(content=f"❌ Configuration Error: {str(ve)}", source="system")
        except Exception as e:
            yield TextMessage(content=f"❌ Error: {str(e)}", source="system")

# Create the literature review system
lit_review_system = LitReviewSystem()

# Main function for Streamlit
async def generate_lit_review(topic: str):
    """Generate literature review with proper error handling"""
    task = f"Conduct a literature review on the topic: {topic}"
    async for message in lit_review_system.run_stream(task=task):
        yield message
