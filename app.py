import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Load LLM
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    openai_api_key=openai_api_key
)

# Load FAISS indexes
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
vectorstore_guidelines = FAISS.load_local("security_faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Create retrievers
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retriever_guidelines = vectorstore_guidelines.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Build QA chains
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
qa_chain_guidelines = RetrievalQA.from_chain_type(llm=llm, retriever=retriever_guidelines, return_source_documents=True)
# 🔹 Analysis function
def analyze_email(email_text, llm, qa_chain, qa_chain_guidelines):
     prompt = (
         "Evaluate whether the following email is a phishing attempt. "
         "Reply with 'Phishing' or 'Not Phishing', and explain your reasoning based on cues like urgency, suspicious links, or sender identity.\n\n"
         f"{email_text}"
     )
     classification = llm.predict(prompt).strip()

     advice_from_emails = qa_chain.invoke(
         f"What are the recommended actions according to cybersecurity best practices if I receive an email like this:\n\n{email_text}"
     )['result']

     advice_from_guidelines = qa_chain_guidelines.invoke(
         f"What are the recommended actions according to cybersecurity best practices if I receive an email like this:\n\n{email_text}"
     )['result']

     return classification, advice_from_emails, advice_from_guidelines


#  🔹 Streamlit UI
st.set_page_config(page_title="Email Phishing Classifier", page_icon="📧", layout="centered")
st.title("📧 Email Phishing Classifier") 
st.write("Paste an email below to check if it’s phishing and get cybersecurity advice.")

#  Input box 
email_text = st.text_area("Email content", height=250, placeholder="Paste your email text here...")

#  Analyze button
if st.button("Analyze Email"):
    if email_text.strip():
        with st.spinner("Analyzing email..."):
            classification, advice_from_emails, advice_from_guidelines = analyze_email(
                email_text, llm, qa_chain, qa_chain_guidelines
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
