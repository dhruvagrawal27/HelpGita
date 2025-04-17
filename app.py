import streamlit as st
from dotenv import load_dotenv
import os
from langchain.chains import RetrievalQA
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("helpgita")


# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Setup Langchain-Pinecone retriever
docsearch = LangchainPinecone(
    index=index,
    embedding=embeddings,
    text_key="text",
    namespace="HelpGita"
)

TEMPLATE = """
You are a helpful and wise guide trained on the Bhagavad Gita, here to answer in simple, easy-to-understand English.  
Use the conversation history and the context below to give clear, practical guidance to the user's question.  

Context from documents:  
{context}  

Current question: {question}  

Please follow this format for your response:  

SOLUTION (do not write this(just for your reference) - Simple Answer):  
Explain the answer in 2–3 short, clear lines that anyone can understand. Use friendly, non-technical English.

ACTION STEPS (do not write this(just for your reference) - Practical & Doable):  
Give 5 specific, real-world steps[do a proper formatting like 1. [Step 1], (next line) 2. [Step 2], (next line) , continue till step 5] the user can actually follow — depending on their question.  
These could include things like exercises, study methods, habits, tools, books, or routines.  
Avoid generic tips — make it personalized and actionable based on the question.
like:
- meditation (with suggested time and mudra if relevant, and a 1-line reason why)
- reading a specific book (with a 1-line reason)
- a breathing exercise or routine
- journaling prompts or reflection
- lifestyle habits with small, meaningful actions  

SOURCE (Gita Reference):  
If possible, mention the exact chapter and verse from the Bhagavad Gita, like: "Bhagavad Gita, Chapter 2, Verse 47".

SHLOK (Original Sanskrit):  
Provide the shlok in Sanskrit with its English transliteration.

EXPLANATION OF CONTEXT (Why this matters):  
Briefly explain when this verse was said in the Gita, what it means, and how it connects to the user’s question.

Response:
"""



PROMPT = PromptTemplate(template=TEMPLATE, input_variables=["chat_history", "context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Load LLM
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.0,
    max_retries=2,
    groq_api_key=GROQ_API_KEY
)

# Build QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Streamlit UI
st.set_page_config(page_title="HelpGita - Spiritual Q&A", page_icon="🕉️")

st.title("🕉️ HelpGita - Ask Your Spiritual Question")
st.markdown("Ask anything about focus, fear, introversion, stress, or purpose based on Bhagavad Gita.")

query = st.text_input("Enter your question:", "")

if st.button("Get Wisdom ✨") and query:
    with st.spinner("Getting spiritual guidance..."):
        try:
            response = qa.invoke({"query": query})
            st.markdown("## 🧘 Answer")
            st.markdown(response["result"])
        except Exception as e:
            st.error(f"Something went wrong: {e}")
