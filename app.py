import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Groq Chatbot 🤖", page_icon="💬", layout="centered")

st.title("💬 Groq-Powered AI Chatbot")
st.write("Chat with an AI model powered by Groq and LangChain!")

# ----------------------------
# Initialize API key
# ----------------------------
if "GROQ_API_KEY" not in os.environ:
    st.error("Missing GROQ_API_KEY. Please set it in Streamlit secrets or .env file.")
    st.stop()

# ----------------------------
# Initialize session state
# ----------------------------
if "chathistory" not in st.session_state:
    st.session_state.chathistory = [
        {"role": "system", "content": "You are a helpful chatbot. Be concise and accurate."}
    ]

# ----------------------------
# Sidebar Settings
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Choose Groq Model:",
        ["llama-3.1-8b-instant", "mixtral-8x7b", "gemma2-9b-it"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    if st.button("Apply"):
        st.session_state["model_name"] = model_name
        st.session_state["temperature"] = temperature
        st.success(f"Model set to {model_name} (temp={temperature})")

# Use updated settings or defaults
model_to_use = st.session_state.get("model_name", "llama-3.1-8b-instant")
temperature_to_use = st.session_state.get("temperature", 0.1)

# ----------------------------
# Initialize LLM and parser
# ----------------------------
llm = ChatGroq(model=model_to_use, temperature=temperature_to_use)
parser = StrOutputParser()

# ----------------------------
# Display Chat History
# ----------------------------
st.subheader("🗨️ Conversation")
for msg in st.session_state.chathistory:
    role, content = msg["role"], msg["content"]
    if role == "user":
        st.chat_message("user").write(content)
    elif role == "assistant":
        st.chat_message("assistant").write(content)
    else:
        st.markdown(f"**System:** {content}")

# ----------------------------
# User Input Section
# ----------------------------
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to history
    st.session_state.chathistory.append({"role": "user", "content": user_input})

    # Build prompt
    prompt = ChatPromptTemplate.from_messages(st.session_state.chathistory)

    # Create chain
    chain = prompt | llm | parser

    # Generate response
    with st.spinner("Thinking..."):
        try:
            response = chain.invoke({})
        except Exception as e:
            st.error(f"Error: {e}")
            response = "Sorry, I encountered an issue."

    # Add response to history
    st.session_state.chathistory.append({"role": "assistant", "content": response})

    # Display the latest assistant message
    st.chat_message("assistant").write(response)
