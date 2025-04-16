import streamlit as st
from dotenv import load_dotenv
import os
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
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

# Prompt template
TEMPLATE = """
Use the following conversation history and context to provide a comprehensive answer to the user's question.  

Context from documents:  
{context}  

Current question: {question}  

Please provide your response in the following format:  

SOLUTION:  
[Provide a 3 line answer]  

STEPS:  
1. [Step 1]  
2. [Step 2]  
3. [Step 3]  
4. [Step 4]  
5. [Step 5]  

SOURCE:  
Tell the exact source for getting to answer [if possible Specify the chapter, verse number, and text name, e.g., "Bhagavad Gita, Chapter 6, Verse 10"]  

SHLOK:  
Tell sanskrit shlok in sanskrit (or hindi language) [Provide the Sanskrit verse with its English transliteration]  

EXPLANATION OF CONTEXT:  
[Explain the context of the shlok—where and in what situation it was said, and how it relates to the current question or problem.]  

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
