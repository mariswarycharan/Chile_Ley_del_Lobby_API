from fastapi import FastAPI, Query, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from typing import List
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

# FastAPI app initialization
app = FastAPI()

# Configure the API key
gemini_api_key = os.environ.get("GEMINI_API_KEY")

# Enable CORS
origins = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://0.0.0.0:8000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add session middleware for user-based chat history
app.add_middleware(SessionMiddleware, secret_key="your-secret-key", session_cookie="session-id")

# Model and embeddings setup
def load_model():
    genai.configure(api_key=gemini_api_key)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return model, embeddings

model, embeddings = load_model()

# Conversational chain setup
def get_conversational_chain():

    system_prompt = """
You are AI ChatBot, an expert assistant specializing in providing detailed and accurate responses strictly based on retrieved information. Your goal is to deliver factual, concise, and helpful answers without introducing speculative or fabricated content.

*INSTRUCTIONS:*
- Base all responses exclusively on the provided context. If the information is not available, clearly state that you do not have enough data to answer.
- Avoid generating information that is not explicitly stated or implied by the retrieved documents.
- Respond politely and informatively.
- Use headings, bullet points, and concise paragraphs for clarity and readability.
- Highlight key points, participants, and outcomes. Avoid over-explaining or speculating beyond the given data.
- Emphasize important actions, follow-ups, and next steps from meetings or discussions.

*ONGOING CONVERSATION:*
The following is a record of the conversation so far, including user queries and assistant responses. Use this to maintain context and provide answers in continuity with previous exchanges.

{chat_history}

*DOCUMENT CONTEXT (if available):*
The following context is retrieved from relevant documents related to the query.

{context}

*USER QUERY:*
{input}

*ASSISTANT RESPONSE:*
Provide a detailed response, keeping prior exchanges in mind. Refer to past questions and answers for continuity. Avoid repeating information unnecessarily but expand on new aspects related to the user's follow-up query.
"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Load FAISS retriever
    new_db = FAISS.load_local("faiss_index", embeddings)
    new_db = new_db.as_retriever(search_type="mmr", search_kwargs={"k": 6})

    # Create chains
    history_aware_retriever = create_history_aware_retriever(model, new_db, prompt)
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

chain = get_conversational_chain()

# API endpoints
@app.get("/")
async def status_check():
    return {"status": "Healthy and running"}

# Pydantic model for input
class InputData(BaseModel):
    user_query: str

@app.post("/get_result")
async def submit_form(data: InputData, request: Request):
    # Retrieve chat history from session
    chat_history = request.session.get("chat_history", [])

    # Process user input
    user_query = data.user_query
    response = chain.invoke(
        {"input": user_query, "context": "Your relevant context here", "chat_history": chat_history}
    )

    # Update chat history in session
    chat_history.append({"user": user_query, "assistant": response["answer"]})
    request.session["chat_history"] = chat_history

    return {"response": response["answer"]}

@app.get("/reset")
async def reset_chat_history(request: Request):
    request.session["chat_history"] = []
    return {"status": "Chat history reset successfully"}
