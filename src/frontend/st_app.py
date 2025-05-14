import streamlit as st
import asyncio
import sys
from pathlib import Path

# Ensure the src directory is in the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Attempt to import run_agent_session
try:
    from src.agents.v1_data_explorer.main import run_agent_session
    from src.config.logger import logger # Optional: if you want to log from streamlit app
except ImportError as e:
    st.error(f"Error importing agent components: {e}. Make sure the agent code is accessible.")
    st.stop()

# Page configuration
st.set_page_config(page_title="Data Explorer Agent Chat", layout="wide")

st.title("💬 Data Explorer Agent")
st.caption("🚀 Interact with the AI agent to explore your data.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If the message is from the assistant and has SQL, show an expander
        if message["role"] == "assistant" and message.get("sql"):
            with st.expander("View Generated SQL"):
                st.code(message["sql"], language="sql")

# Handle new user input
if prompt := st.chat_input("Ask the agent..."):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with the agent
    with st.chat_message("assistant"):
        with st.spinner("🕵️‍♂️ Agent is thinking..."):
            try:
                # Run the synchronous function that internally handles asyncio
                agent_response_data = asyncio.run(run_agent_session(prompt))
                
                agent_response = agent_response_data.get("response_content", "Error: No response content found.")
                generated_sql = agent_response_data.get("generated_sql")

                # Ensure agent_response is a string
                if not isinstance(agent_response, str):
                    logger.error(f"Agent response content was not a string: {type(agent_response)} - {agent_response}")
                    agent_response = "Sorry, I received an unexpected response format from the agent."
                
            except Exception as e:
                logger.error(f"Error during agent session: {e}", exc_info=True)
                agent_response = f"An error occurred: {e}"
                generated_sql = None # Ensure sql is None on error
            
            st.markdown(agent_response)
            # If SQL was generated, show the expander immediately for the latest response
            if generated_sql:
                with st.expander("View Generated SQL", expanded=False): # Can set expanded=True if you want it open by default
                    st.code(generated_sql, language="sql")

    # Add assistant message to chat history, including SQL if any
    st.session_state.messages.append({
        "role": "assistant", 
        "content": agent_response,
        "sql": generated_sql
    })

# For debugging: Show session state
# st.sidebar.subheader("Session State")
# st.sidebar.write(st.session_state)
