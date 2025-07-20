import streamlit as st
import asyncio
from agent_be import generate_lit_review
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure page
st.set_page_config(
    page_title="AutoGen Literature Review", 
    page_icon="📚",
    layout="wide"
)

# Header
st.title("📚 ScholarSynth AI — Multi-Agent Literature Review Assistant")
st.markdown("*Powered by AutoGen Multi-Agent System with Groq API*")

# Try to get API key from multiple sources
def get_api_key():
    # First try environment variable (most common for local dev)
    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key:
        return env_key
    
    # Then try Streamlit secrets (for deployment)
    try:
        if hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    
    return ""

# Initialize session state for API key
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = get_api_key()

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Key input
    st.subheader("🔑 Groq API Key")
    
    # Check if API key exists in environment
    env_api_key = os.getenv("GROQ_API_KEY")
    if env_api_key:
        st.success("✅ API Key loaded from environment")
        st.session_state.groq_api_key = env_api_key
    else:
        # Allow user to input API key
        api_key_input = st.text_input(
            "Enter your Groq API Key:",
            type="password",
            value=st.session_state.groq_api_key,
            help="Get your free API key from: https://console.groq.com/keys"
        )
        
        if api_key_input:
            st.session_state.groq_api_key = api_key_input
            # Set environment variable for the current session
            os.environ["GROQ_API_KEY"] = api_key_input
            st.success("✅ API Key set for this session")
        else:
            st.warning("⚠️ Please enter your Groq API key to use the application")
    
    st.markdown("---")
    st.subheader("📋 System Info")
    st.markdown("""
    **Multi-Agent Workflow:**
    1. 🔍 **Researcher Agent** - Fetches papers from arXiv
    2. 📝 **Summarizer Agent** - Creates literature review
    
    **Note:** Get your free Groq API key from [console.groq.com](https://console.groq.com/keys)
    """)

# Main interface
st.subheader("🔍 Research Topic Input")

col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "Enter Research Topic:", 
        placeholder="e.g., transformer neural networks, reinforcement learning, quantum computing",
        help="Enter any academic research topic",
        label_visibility="collapsed"
    )

with col2:
    st.write("") # spacing
    # Check if API key is available
    has_api_key = bool(st.session_state.groq_api_key.strip())
    run_btn = st.button(
        "🚀 Generate Review", 
        type="primary", 
        disabled=not has_api_key,
        help="Enter API key first" if not has_api_key else "Generate literature review"
    )

# Show API key status
if not has_api_key:
    st.error("🔑 Please enter your Groq API key in the sidebar to continue")
    st.info("💡 You can get a free API key from [console.groq.com](https://console.groq.com/keys)")

# Results section
if run_btn and query and has_api_key:
    if not query.strip():
        st.error("Please enter a research topic.")
    else:
        # Create containers for real-time updates
        status_container = st.empty()
        content_container = st.container()
        
        try:
            # Show loading state
            with status_container:
                st.info("🤖 AutoGen agents are working... This may take 30-60 seconds")
                progress_bar = st.progress(0)
                # Simple progress animation without await
                import time
                for i in range(100):
                    progress_bar.progress(i + 1)
                    time.sleep(0.001)  # Small delay for progress animation
            
            # Collect all messages
            messages = []
            
            # Run the async generator
            async def collect_messages():
                message_count = 0
                async for msg in generate_lit_review(query):
                    messages.append(msg)
                    message_count += 1
                    
                    # Update display in real-time
                    with content_container:
                        if message_count == 1:
                            st.subheader("📄 Generated Literature Review")
                        
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
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        st.download_button(
                            label="📥 Download as Markdown",
                            data=final_content,
                            file_name=f"literature_review_{query.replace(' ', '_')}.md",
                            mime="text/markdown"
                        )
                    with col2:
                        # Convert to plain text for .txt download
                        import re
                        plain_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', final_content)  # Remove markdown links
                        plain_text = re.sub(r'[#*`]', '', plain_text)  # Remove markdown formatting
                        st.download_button(
                            label="📄 Download as Text",
                            data=plain_text,
                            file_name=f"literature_review_{query.replace(' ', '_')}.txt",
                            mime="text/plain"
                        )
                    
        except Exception as e:
            status_container.empty()
            error_msg = str(e)
            if "GROQ_API_KEY" in error_msg:
                st.error("🔑 API Key Error: Please check your Groq API key")
                st.info("💡 Make sure your API key is valid and has sufficient quota")
            elif "rate limit" in error_msg.lower():
                st.error("⏱️ Rate Limit: Too many requests. Please wait a moment and try again")
            elif "timeout" in error_msg.lower():
                st.error("⏱️ Timeout: Request took too long. Please try again")
            else:
                st.error(f"❌ Error: {error_msg}")
                st.info("Please check your internet connection and try again")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p><em>Built with AutoGen, Groq API, and arXiv • Open Source Literature Review Assistant</em></p>
        <p><small>⭐ If you find this useful, consider starring the project on GitHub!</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)
