import streamlit as st
import requests

st.set_page_config(page_title="RAG Agent", layout="wide")

st.title("🤖 文档分析 Agent")

if "history" not in st.session_state:
    st.session_state.history = []

# ===== 输入框 =====
query = st.chat_input("请输入你的问题...")

if query:
    st.session_state.history.append(("user", query))

    # 调用后端
    try:
        resp = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"query": query},
            timeout=None
        )
        answer = resp.json()["answer"]
    except Exception as e:
        answer = f"❌ 请求失败: {e}"

    st.session_state.history.append(("assistant", answer))


# ===== 显示聊天 =====
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)