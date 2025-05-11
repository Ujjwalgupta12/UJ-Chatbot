from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

template = """
Answer the question below

Here is the conversation history : {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

context = ""

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message")
    global context

    result = chain.invoke({"context": context, "question": user_input})
    context += f"\nUser: {user_input}\nUJ: {result}"
    return {"response": result}
