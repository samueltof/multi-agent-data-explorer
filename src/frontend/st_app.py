import streamlit as st
import asyncio
import sys
import re # Import re for the formatter
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

# --- Simple SQL Formatter ---
def minimal_sql_formatter(sql_string: str) -> str:
    if not sql_string or not isinstance(sql_string, str):
        return str(sql_string) # Ensure it's a string if not None or already string

    original_sql = sql_string.strip()

    # If it already has newlines, assume it's somewhat formatted by the agent.
    if '\n' in original_sql:
        return original_sql

    # For single-line SQL, add newlines before major keywords
    formatted_sql = original_sql

    # Keywords that should start a new line
    newline_keywords = [
        "FROM", "LEFT OUTER JOIN", "RIGHT OUTER JOIN", "FULL OUTER JOIN", 
        "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "JOIN", 
        "WHERE", "GROUP BY", "ORDER BY", "LIMIT", "HAVING", 
        "UNION ALL", "UNION", 
        "VALUES", "SET", "ON"
    ]

    # Handle SELECT clause indentation first
    select_prefix = "SELECT"
    select_prefix_len = len(select_prefix)
    if formatted_sql.upper().startswith(select_prefix):
        # Find where the column list ends (heuristic: before FROM)
        from_keyword_pos = -1
        match_from = re.search(r"\sFROM\s", formatted_sql, re.IGNORECASE)
        if match_from:
            from_keyword_pos = match_from.start()
        
        # Extract parts
        if from_keyword_pos != -1:
            columns_part_str = formatted_sql[select_prefix_len:from_keyword_pos].strip()
            rest_of_sql = formatted_sql[from_keyword_pos:] # Includes leading space before FROM
            
            columns = [col.strip() for col in columns_part_str.split(',')]
            if len(columns) > 1:
                indented_columns = "  " + ",\n  ".join(columns)
                formatted_sql = f"{select_prefix}\n{indented_columns}{rest_of_sql}"
            elif columns_part_str: # Single column or complex expression
                formatted_sql = f"{select_prefix}\n  {columns_part_str}{rest_of_sql}"
            # If columns_part_str is empty (e.g. SELECT *), it will just be SELECT plus rest_of_sql
            # The existing logic should handle this by not changing formatted_sql if columns_part_str is empty.
    
    # For other keywords, insert a newline before them.
    # This preserves the original casing of the keyword found.
    for kw in newline_keywords:
        pattern = r"(\s)(\b" + re.escape(kw) + r"\b)" # Space, then keyword
        def replace_with_newline(match_obj):
            return f"\n{match_obj.group(2)}" # Newline + original keyword
        formatted_sql = re.sub(pattern, replace_with_newline, formatted_sql, flags=re.IGNORECASE)

    return formatted_sql.strip()
# --- End of SQL Formatter ---

# Page configuration
st.set_page_config(page_title="Data Explorer Agent Chat", layout="wide")

st.title("💬 Data Explorer Agent")
st.caption("🚀 Interact with the AI agent to explore your data.")

# --- Starter Questions Sidebar ---
st.sidebar.title("💡 Starter Questions")
st.sidebar.markdown("Click a question to ask the agent:")

starter_questions = [
    "How many TCR complexes are in the database?",
    "How many alpha versus beta chains do we have in total?",
    "Which epitopes are represented, and how many complexes recognize each?",
    "Which V-gene segments are most frequently used overall in the dataset?"
]

if "submitted_starter_query" not in st.session_state:
    st.session_state.submitted_starter_query = None

for i, question in enumerate(starter_questions):
    if st.sidebar.button(question, key=f"starter_{i}"):
        st.session_state.submitted_starter_query = question
        st.rerun() # Rerun to process the selected query

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
                formatted_sql_display = minimal_sql_formatter(message["sql"])
                st.code(formatted_sql_display, language="sql")

# Determine the prompt: either from chat_input or a submitted starter query
user_prompt = None
chat_input_prompt = st.chat_input("Ask the agent...")

if st.session_state.submitted_starter_query:
    user_prompt = st.session_state.submitted_starter_query
    st.session_state.submitted_starter_query = None # Clear after use
elif chat_input_prompt:
    user_prompt = chat_input_prompt

# Handle new user input (either from chat_input or starter question)
if user_prompt:
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Process with the agent
    with st.chat_message("assistant"):
        with st.spinner("🕵️‍♂️ Agent is thinking..."):
            try:
                # Run the synchronous function that internally handles asyncio
                agent_response_data = asyncio.run(run_agent_session(user_prompt))
                
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
                    formatted_sql_display = minimal_sql_formatter(generated_sql)
                    st.code(formatted_sql_display, language="sql")

    # Add assistant message to chat history, including SQL if any
    st.session_state.messages.append({
        "role": "assistant", 
        "content": agent_response,
        "sql": generated_sql
    })

# For debugging: Show session state
# st.sidebar.subheader("Session State")
# st.sidebar.write(st.session_state)
