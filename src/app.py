import streamlit as st
import uuid
import json
import os
from datetime import datetime

# ✅ FIX: import module instead of function
import generate

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Chatbot", layout="wide")

# Project root, whichever folder streamlit is started from
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chats.json")

# ---------------- CACHE BACKEND ----------------
@st.cache_resource
def get_generator():
    # This ensures your model + vector DB load ONLY once
    return generate

# ---------------- STORAGE ----------------
def load_chats():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_chats(chats):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, indent=2)

# ---------------- STATE INIT ----------------
if "chats" not in st.session_state:
    st.session_state.chats = load_chats()

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# ---------------- HELPERS ----------------
def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "created_at": str(datetime.now())
    }
    st.session_state.current_chat_id = chat_id
    save_chats(st.session_state.chats)

def delete_chat(chat_id):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        if st.session_state.current_chat_id == chat_id:
            st.session_state.current_chat_id = None
        save_chats(st.session_state.chats)

def set_chat_title(chat_id, first_message):
    title = first_message[:40] + ("..." if len(first_message) > 40 else "")
    st.session_state.chats[chat_id]["title"] = title

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("💬 Chats")

    if st.button("➕ New Chat"):
        create_new_chat()

    st.markdown("---")

    for chat_id, chat in list(st.session_state.chats.items()):
        col1, col2 = st.columns([4, 1])

        with col1:
            if st.button(chat["title"], key=chat_id):
                st.session_state.current_chat_id = chat_id

        with col2:
            if st.button("🗑️", key=f"del_{chat_id}"):
                delete_chat(chat_id)
                st.rerun()  # redraw the list without the deleted chat

# ---------------- MAIN CHAT ----------------
st.title("AI Assistant")

if st.session_state.current_chat_id is None:
    st.info("Start a new chat from the sidebar.")
    st.stop()

chat = st.session_state.chats[st.session_state.current_chat_id]

# ---------------- CHAT DISPLAY ----------------
# st.chat_message renders the text as markdown with HTML escaped, so code like
# "List<int>" or "a < b" shows correctly and line breaks are kept (the old raw
# HTML divs swallowed both)
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- INPUT + MESSAGE HANDLING ----------------
user_input = st.chat_input("Type your message...")

if user_input and user_input.strip():
    # Add user message
    chat["messages"].append({"role": "user", "content": user_input})

    # Set title if first message
    if len(chat["messages"]) == 1:
        set_chat_title(st.session_state.current_chat_id, user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    # ✅ Use cached backend
    gen = get_generator()

    # Show thinking
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = gen.generate_answer(user_input)
        st.markdown(answer)

    # Add assistant message
    chat["messages"].append({"role": "assistant", "content": answer})

    save_chats(st.session_state.chats)

    # Rerun so the sidebar shows the new chat title
    st.rerun()