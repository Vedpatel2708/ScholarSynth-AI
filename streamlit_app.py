import streamlit as st
import asyncio
from agent_be import generate_lit_review
import os

# Configure page
st.set_page_config(
    page_title="AutoGen Literature Review", 
    page_icon="📚",
    layout="wide"
)

# Header
st.title("📚 ScholarSynth AI — Multi-Agent Literature Review Assistant")
st.markdown("*Powered by AutoGen Multi-Agent System with Groq API*")

# Sidebar
with st.sidebar:
    st.header("🔧 System Info")
    st.markdown("""
    **Multi-Agent Workflow:**
    1. 🔍 **Researcher Agent** - Fetches papers from arXiv
    2. 📝 **Summarizer Agent** - Creates literature review
    """)
    
    # Check API key
    api_key_status = "✅ Set" if os.getenv("GROQ_API_KEY") else "❌ Missing"
    st.markdown(f"**Groq API Key:** {api_key_status}")
    
    if not os.getenv("GROQ_API_KEY"):
        st.error("Please set GROQ_API_KEY environment variable")

# Main interface
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "🔍 Enter Research Topic:", 
        placeholder="e.g., transformer neural networks, reinforcement learning, quantum computing",
        help="Enter any academic research topic"
    )

with col2:
    st.write("") # spacing
    run_btn = st.button("🚀 Generate Review", type="primary", disabled=not os.getenv("GROQ_API_KEY"))

# Results section
if run_btn and query:
    if not query.strip():
        st.error("Please enter a research topic.")
    else:
        # Create containers for real-time updates
        status_container = st.empty()
        content_container = st.container()
        
        try:
            # Show loading state
            with status_container:
                st.info("🤖 AutoGen agents are working...")
            
            # Collect all messages
            messages = []
            
            # Run the async generator
            async def collect_messages():
                async for msg in generate_lit_review(query):
                    messages.append(msg)
                    
                    # Update display in real-time
                    with content_container:
                        current_content = ""
                        for m in messages:
                            if hasattr(m, 'content'):
                                current_content += m.content + "\n\n"
                        
                        if current_content.strip():
                            st.markdown(current_content)
            
            # Run async function
            asyncio.run(collect_messages())
            
            # Clear status
            status_container.empty()
            
            # Add download button if we have content
            if messages:
                final_content = "\n\n".join([m.content for m in messages if hasattr(m, 'content')])
                if final_content.strip():
                    st.download_button(
                        label="📥 Download Review",
                        data=final_content,
                        file_name=f"literature_review_{query.replace(' ', '_')}.md",
                        mime="text/markdown"
                    )
                    
        except Exception as e:
            status_container.empty()
            st.error(f"Error: {str(e)}")
            st.info("Please check your internet connection and API key.")

# Footer
st.markdown("---")
st.markdown("*Built with AutoGen, Groq API, and arXiv*")