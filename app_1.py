import streamlit as st
from query import get_formatted_response, rag_chain

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="LawGuide AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Sidebar ----
with st.sidebar:
    st.title("📚 Explore Law Domains")
    st.markdown("Choose your area of interest:")
    st.checkbox("📜 Constitution", value=True)
    st.checkbox("⚖️ Criminal Law")
    st.checkbox("💼 Property Law")
    st.checkbox("🏛️ Civil Law")
    st.markdown("---")
    st.info("🔐 Your queries are private and not stored.")

# ---- Header ----
st.markdown("## Welcome to **LawGuide AI**")
st.markdown("""
Get instant, document-backed legal explanations from the Indian Constitution and more.  
""")

st.markdown("---")

# ---- Query Section ----
st.markdown("### 🔍 Ask a Legal Question")
user_query = st.text_input("Enter your legal question here...", placeholder="e.g., What does Article 21 guarantee?")
show_sources = st.toggle("🔎 Show Context Used", value=False)

# ---- Answer Button ----
if st.button("📨 Get Answer"):
    if not user_query.strip():
        st.warning("⚠️ Please enter a valid legal question.")
    else:
        with st.spinner("🔍 Searching legal documents..."):
            result = get_formatted_response(user_query)
            st.markdown("### ✅ AI Legal Assistant's Response")
            st.markdown(result["formatted_answer"], unsafe_allow_html=True)

            if show_sources:
                docs = rag_chain.retriever.get_relevant_documents(user_query)
                if docs:
                    st.markdown("### 📖 Retrieved Context")
                    for i, doc in enumerate(docs):
                        with st.expander(f"Chunk {i+1}"):
                            st.code(doc.page_content.strip()[:800], language="markdown")
                else:
                    st.info("No specific context found.")

# ---- Featured Law Highlights ----
st.markdown("---")
st.markdown("### 🌟 Explore Key Constitutional Articles")

cols = st.columns(4)
topics = [
    ("🔐 Article 21", "Right to life and personal liberty."),
    ("🛕 Article 25", "Freedom of religion."),
    ("📄 Article 14", "Equality before law."),
    ("📊 Article 368", "Amendment procedure.")
]
for col, (title, desc) in zip(cols, topics):
    with col:
        st.markdown(f"#### {title}")
        st.info(desc)

# ---- Feedback Box ----
st.markdown("---")
st.markdown("### 💬 Feedback")
st.text_area("What do you think about LawGuide AI? Suggest features or report issues.")

# ---- Footer ----
st.markdown("---")
st.markdown(
    "<div style='text-align: center; padding: 10px; font-size: 14px;'>"
    "⚖️ Built using LangChain, HuggingFace, FAISS, and Groq LLaMA3 | Designed by Shiva Prasad"
    "</div>",
    unsafe_allow_html=True
)
