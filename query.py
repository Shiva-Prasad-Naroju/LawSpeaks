"""
Indian Constitution Legal QA System with RAG

- Uses HuggingFace embeddings + FAISS for context retrieval.
- Groq llama3-70b-8192 for accurate, detailed answer generation.
- Retrieval k=4 selected as best trade-off for precision & context richness.
- Prompt strictly enforces using ONLY provided context with no speculation.
- Supports partial answers with explicit mention of missing info.
- Avoids hallucinations and mislabeling of constitutional Articles.
- Designed for clarity and simplicity in legal explanations.
"""

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os
load_dotenv()

# Set API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

try:
    vectorstore = FAISS.load_local("vector_db_1", embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS vectorstore: {e}")


retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  
# Top-K Comparison Notes:
# Tried k = 2, 4, 8, 12 with LLaMA3-70B-8192 to test RAG response quality.
# - k=2: Most concise, less detailed, misses some context
# - k=4: Balanced – good factual accuracy + context relevance
# - k=8: More detailed but starts to overfit or repeat
# - k=12: Verbose, sometimes includes redundant or less relevant info

# Verdict: k=4 provides the best balance of accuracy, relevance, and clarity.

# Load LLM
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model="llama3-70b-8192") # slow but more accurate

# Define powerful legal prompt
prompt = ChatPromptTemplate.from_template("""
You are a highly accurate legal assistant specialized in Indian constitutional law.

Use **only** the information provided in the <context> block to answer the user's legal question.

If a specific Article, Clause, Entry, or Schedule is mentioned in the context, **quote it accurately** first, then explain it in clear and simple terms.

If Article numbers are **only indirectly referenced**, carefully infer and explain the most likely related Article **based solely on the legal phrasing in the context**.

If the question is **partially answered** in the context, clearly state what is explicitly present and what is missing or not found.

⚠️ Do not guess or fabricate legal content.
⚠️ Do not mislabel Article numbers.
⚠️ If the answer is not found in the context, respond exactly with:  
**"Not found in the provided documents."**

<context>
{context}
</context>

Question: {input}

Answer:
Provide a detailed, clear, and well-structured answer using complete sentences. Explain legal terms simply and provide context or implications where relevant. Avoid very short or one-line answers.
""")

# Create chains
doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, doc_chain)

# ✅ Expose only rag_chain for import
__all__ = ["rag_chain"]



def get_formatted_response(user_question: str) -> dict:
    """
    Runs the RAG chain and formats the response in a structured, user-friendly format.
    Returns a dictionary with `formatted_answer` and `raw_answer`.
    """
    response = rag_chain.invoke({"input": user_question})
    raw_answer = response.get("answer", "").strip()

    # Handle fallback case
    if raw_answer == "Not directly mentioned provided dataset.":
        formatted = f"""
### 🔍 Question Asked:
**{user_question}**

---

### 📄 Based on the documents reviewed:

- The retrieved documents relate to other constitutional provisions.
- However, **they do not contain specific information about your query**.
- No matching Article, Clause, or related law was identified in this context.

---

📌 *Tip: Try rephrasing your query or narrowing it down to a specific Article or keyword for better results.*
        """
    else:
        # Standard answer, return as-is but with headers
        formatted = f"""
### 🔍 Question Asked:
**{user_question}**

---

### ✅ AI Legal Assistant's Response:
{raw_answer}
        """

    return {
        "formatted_answer": formatted.strip(),
        "raw_answer": raw_answer
    }

