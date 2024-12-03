from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Use the environment variable directly
api_key = os.getenv("GROQ_API_KEY")

# Initialize ChatGroq with the API key
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    api_key=api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers the user's questions in a concise manner.If you do not understand the user please ask for further clarification.",
        ),
        ("human", "{input}"),
    ]
)

# Chatting in a loop
while True:
    input_text = input("Enter text to chat with me! (or type 'exit' to quit) : ")
    if input_text.lower() == "exit":
        break
    
    # Invoke the chat model
    chain = prompt | llm
    aimsg = chain.invoke(
        {
            "input": input_text
        }
    )
    
    print("AI : ", aimsg.content)
