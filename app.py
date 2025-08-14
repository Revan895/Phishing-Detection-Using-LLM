import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from pydantic import BaseModel, Field

# 🔐 API Key (consider loading from .env for security)
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 🔹 Load LLM
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    openai_api_key=openai_api_key
)

# 🔹 Load Embedding Model
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# 🔹 Load FAISS Indexes
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
vectorstore_guidelines = FAISS.load_local("security_faiss_index", embedding_model, allow_dangerous_deserialization=True)

# 🔹 Create Raw Retrievers
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retriever_guidelines = vectorstore_guidelines.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 🔹 Label-Aware Retriever Wrapper
class LabelInjectingRetriever(BaseRetriever, BaseModel):
    base_retriever: BaseRetriever = Field(...)

    def get_relevant_documents(self, query):
        return self.base_retriever.get_relevant_documents(query)

    async def aget_relevant_documents(self, query):
        return self.get_relevant_documents(query)

# 🔹 Format Retrieved Docs with Labels
def format_docs_with_labels(docs):
    parts = []
    for doc in docs:
        label = doc.metadata.get("label", "Unknown")
        parts.append(f"[Label: {label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

# 🔹 Wrap Retrievers
wrapped_retriever = LabelInjectingRetriever(base_retriever=retriever)
wrapped_retriever_guidelines = LabelInjectingRetriever(base_retriever=retriever_guidelines)

# 🔹 Build QA Chains
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=wrapped_retriever, return_source_documents=True)
qa_chain_guidelines = RetrievalQA.from_chain_type(llm=llm, retriever=wrapped_retriever_guidelines, return_source_documents=True)

# 🔹 Email Analysis Function
def analyze_email(email_text, llm, wrapped_retriever, qa_chain, qa_chain_guidelines):
    similar_docs = wrapped_retriever.get_relevant_documents(email_text)
    labeled_similar_emails = format_docs_with_labels(similar_docs)

    prompt_text = (
        "You are a cybersecurity expert. Here are some similar emails with their phishing labels:\n\n"
        f"{labeled_similar_emails}\n\n"
        "Please carefully analyze these examples and their labels.\n"
        "Using that information, classify the following new email as 'Phishing' or 'Not Phishing'.\n"
        "In your explanation, explicitly reference how the examples influenced your decision.\n\n"
        f"New email:\n{email_text}"
    )

    classification = llm.predict(prompt_text).strip()

    advice_prompt = (
        f"What are the recommended actions according to cybersecurity best practices "
        f"if I receive an email like this? If it's not a phishing email, don't give any advice:\n\n{email_text}"
    )

    advice_from_emails = qa_chain.invoke(advice_prompt)['result']
    advice_from_guidelines = qa_chain_guidelines.invoke(advice_prompt)['result']

    return classification, advice_from_emails, advice_from_guidelines, similar_docs

# 🔹 Streamlit UI
st.set_page_config(page_title="Email Phishing Classifier", page_icon="📧", layout="centered")
st.title("📧 Email Phishing Classifier")
st.write("Paste an email below to check if it’s phishing and get cybersecurity advice.")

email_text = st.text_area("Email content", height=250, placeholder="Paste your email text here...")

if st.button("Analyze Email"):
    if email_text.strip():
        with st.spinner("Analyzing email..."):
            classification, advice_from_emails, advice_from_guidelines, _ = analyze_email(
                email_text, llm, wrapped_retriever, qa_chain, qa_chain_guidelines
            )

        st.success("Analysis Complete ✅")

        st.subheader("📌 Classification")
        st.write(classification)

        st.subheader("📝 Advice from Similar Emails")
        st.write(advice_from_emails)

        st.subheader("📚 Advice from Security Guidelines")
        st.write(advice_from_guidelines)
    else:
        st.warning("Please paste an email to analyze.")