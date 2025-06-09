import streamlit as st
import requests
import base64

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ContentBot",
    page_icon="📈",
    layout="wide"
)

# --- CONFIGURATION ---
# Load the backend URL from Streamlit Secrets
try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except (FileNotFoundError, KeyError):
    st.warning("BACKEND_URL secret not found.")
    BACKEND_URL = "http://127.0.0.1:8000" # Fallback for local testing

# --- HELPER FUNCTIONS ---
def display_base64_image(base64_string):
    """Decodes and displays a base64 encoded image."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        st.image(base64.b64decode(base64_string), use_container_width=True)
    except Exception as e:
        st.error(f"Could not display image: {e}")

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("About")
    st.info("This chatbot is an expert on the 'Price Trends' document. Ask any questions about its content.")
    
    st.divider()

    st.header("Chat Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Chat cleared. How can I help?"}]
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("🤖CustomGPT-OriginBluy")
st.markdown("Hello... feel free to drop your query")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you analyze the Price Trends report?"}]

# (The rest of your chat interface loop for displaying and handling messages remains exactly the same)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            with st.expander("Show Sources"):
                for text_source in message["sources"].get("texts", []):
                    st.info("Retrieved Text Snippet:")
                    st.text(text_source)
                for img_source in message["sources"].get("images", []):
                    st.info("Retrieved Image:")
                    display_base64_image(img_source)

if prompt := st.chat_input("Ask about CPI, WPI, inflation trends..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                query_url = f"{BACKEND_URL}/query/"
                response = requests.post(query_url, json={"question": prompt})
                response.raise_for_status()
                
                data = response.json()
                answer = data.get("answer", "I couldn't generate an answer.")
                st.write(answer)
                
                assistant_message = {"role": "assistant", "content": answer, "sources": data}
                st.session_state.messages.append(assistant_message)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
    st.rerun()