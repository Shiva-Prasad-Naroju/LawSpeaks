import streamlit as st
from query import rag_chain  
from langchain_community.vectorstores import FAISS

# ---- Streamlit Page Config ----
st.set_page_config(page_title="LawGuide", page_icon="⚖️", layout="centered")
st.title("🧠 LawGuide – Your Legal AI Assistant")
st.markdown("Ask questions from the Indian Constitution, IPC, CrPC and get reliable answers based **only on authentic documents.**")

# ---- Input Section ----
query = st.text_input("📌 Enter your legal query:")
show_sources = st.toggle("Show Source Context", value=False)

# ---- Ask Button ----
if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a valid legal question.")
    else:
        with st.spinner("🔍 Searching the Constitution..."):
            response = rag_chain.invoke({"input": query})
            answer = response.get("answer", "⚠️ No answer generated.")
            st.success("✅ Answer:")
            st.write(answer)

        if show_sources:
            st.markdown("---")
            st.subheader("📖 Retrieved Source Context")
            context = response.get("context", None)
            if context:
                st.code(context, language="markdown")
            else:
                st.info("No specific context found or context display not enabled in your chain.")


# ---- Footer ----
st.markdown("---")
st.markdown("⚖️ *Powered by FAISS + HuggingFace + LLaMA 3 (Groq)*")
