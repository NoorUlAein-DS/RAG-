import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# 1. Page Configuration
st.set_page_config(
    page_title="JUW AI Agent",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS for Modern Chat UI & JUW Theme Colors (Green & Blue Gradient)
st.markdown("""
    <style>
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Styling with JUW Green & Blue Gradient */
    .chat-header {
        background: linear-gradient(135deg, #0A5C36 0%, #1E3A8A 100%);
        padding: 20px 24px;
        border-radius: 14px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .header-title {
        font-size: 22px;
        font-weight: 700;
        margin: 0 0 4px 0;
        letter-spacing: 0.5px;
    }
    .header-subtitle {
        font-size: 13px;
        color: #E2E8F0;
        margin: 0;
    }
    /* Tag Pills Styling */
    .tag-container {
        display: flex;
        gap: 8px;
        margin-top: 10px;
        flex-wrap: wrap;
    }
    .tag-pill {
        background: rgba(255, 255, 255, 0.18);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 500;
        color: #FFFFFF;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stChatMessage {
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Header Section (Cap Icon removed, clean aesthetic with university badge tags)
st.markdown("""
    <div class="chat-header">
        <div>
            <h3 class="header-title">JUW AI Agent</h3>
            <p class="header-subtitle">Jinnah University for Women - Official Admissions & Campus Assistant</p>
            <div class="tag-container">
                <span class="tag-pill">🏛️ Admissions</span>
                <span class="tag-pill">📚 Courses</span>
                <span class="tag-pill">👩‍🏫 Faculty</span>
            </div>
        </div>
        <div style="font-size: 32px; background: rgba(255,255,255,0.15); padding: 10px 14px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.2);">🏛️</div>
    </div>
""", unsafe_allow_html=True)

# 4. Load Vector Store & Groq Client (Fixed path for Streamlit Cloud deployment)
@st.cache_resource
def initialize_rag():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db_path = "faiss_index"
    
    if not os.path.exists(db_path):
        raise FileNotFoundError("The 'faiss_index' folder is missing. Please ensure it is uploaded to your GitHub repository.")
        
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

try:
    vectorstore = initialize_rag()
except Exception as e:
    st.error(f"Database loading error: {e}")
    st.info("Please make sure the 'faiss_index' folder (containing index.faiss and index.pkl) is pushed to your GitHub repository alongside app.py.")
    st.stop()

try:
    groq_api_key = st.secrets["GroqApiKey"]
except:
    groq_api_key = os.getenv("GroqApiKey")

client = Groq(api_key=groq_api_key)

# 5. System Prompt Setup (Strictly English)
SYSTEM_PROMPT = """
You are the official AI assistant for Jinnah University for Women (JUW).
Answer user queries accurately using ONLY the provided JUW context.

STRICT LANGUAGE RULES:
1. Respond ONLY in clear, professional English.
2. Do NOT use Roman Urdu, Urdu script, or any other language under any circumstances.
3. If information is not in the context, respond strictly with:
'Information regarding this specific query is currently not available in my official JUW database.'
"""

# 6. Initialize Chat History & First Welcome Message
if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome_msg = "Hello! I am the official JUW AI Assistant. How can I help you with Jinnah University for Women admissions, courses, faculty, or campus details today?"
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🎓" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])

# 7. Handle User Input
if user_query := st.chat_input("Ask anything about JUW admissions, courses, or faculty..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Searching JUW database..."):
            try:
                docs = vectorstore.similarity_search(user_query, k=10)
                context = "\n\n".join([doc.page_content for doc in docs])

                messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContext:\n{context}"}]
                for m in st.session_state.messages:
                    messages.append({"role": m["role"], "content": m["content"]})

                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.0
                )
                bot_reply = response.choices[0].message.content.strip()
                
                st.markdown(bot_reply)
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                
            except Exception as e:
                error_reply = f"An error occurred: {str(e)}"
                st.markdown(error_reply)
                st.session_state.messages.append({"role": "assistant", "content": error_reply})
