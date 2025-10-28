import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# ---------------------------
# Initialize model & parser
# ---------------------------
# Create a single model instance to reuse across requests
model = OllamaLLM(model="gemma3:4b", temperature=0.1)
parser = StrOutputParser()

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="Generative Ai chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Generative AI Chatbot")
st.write("A simple chat interface using LangChain's ChatPromptTemplate and Ollama LLM.")

# ---------------------------
# Initialize session state
# ---------------------------
if "chathistory" not in st.session_state:
    st.session_state.chathistory = [
        {"role": "system", "content": "You are a helpful chatbot. Be concise and accurate."}
    ]

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    model_name = st.text_input("Ollama model", value="gemma3:4b")
    if st.button("Apply"):
        # update model (recreate instance)
        st.session_state["model"] = OllamaLLM(model=model_name, temperature=temperature)
        st.success(f"Model set to {model_name} with temp={temperature}")

# If user changed model via sidebar, use that instance; otherwise reuse the global one
llm = st.session_state.get("model", model)

# ---------------------------
# Chat display
# ---------------------------
chat_area = st.container()
with chat_area:
    st.subheader("Conversation")
    for message in st.session_state.chathistory:
        role = message.get("role")
        content = message.get("content")
        # streamlit has a convenient chat message UI element; fall back to markdown if not available
        try:
            if role == "user":
                st.chat_message("user").write(content)
            elif role == "assistant":
                st.chat_message("assistant").write(content)
            else:
                st.markdown(f"**System:** {content}")
        except Exception:
            # fallback for older streamlit versions
            if role == "user":
                st.markdown(f"**You:** {content}")
            elif role == "assistant":
                st.markdown(f"**Bot:** {content}")
            else:
                st.markdown(f"**System:** {content}")

st.markdown("---")

# ---------------------------
# User input area
# ---------------------------
with st.form("user_input_form", clear_on_submit=True):
    user_text = st.text_input("You:", key="input")
    submitted = st.form_submit_button("Send")

if submitted and user_text:
    # append user message to history
    st.session_state.chathistory.append({"role": "user", "content": user_text})

    # build prompt from messages
    prompt = ChatPromptTemplate.from_messages(st.session_state.chathistory)

    # create chain and run (prompt -> llm -> parser)
    chain = prompt | llm | parser

    # invoke chain; chain.invoke takes a mapping if the prompt needs variables.
    # Since from_messages fully defines the prompt, we pass an empty dict.
    try:
        response = chain.invoke({})
    except Exception as e:
        st.error(f"Model invocation failed: {e}")
        response = "Sorry, I couldn't generate a response."

    # append assistant response to history
    st.session_state.chathistory.append({"role": "assistant", "content": response})

    # Rerun the app to show new messages (Streamlit automatically reruns, but we ensure UI updates)
st.rerun()

