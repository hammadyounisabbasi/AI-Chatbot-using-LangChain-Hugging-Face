from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
import streamlit as st  # run streamlit in terminal using "streamlit run 2_prompt_ui.py"
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()
api_token = os.getenv("HUGGINGFACE_API_TOKEN")

model_repo = "meta-llama/Llama-3.1-8B-Instruct"  

llm = HuggingFaceEndpoint(
    repo_id=model_repo,
    task="text-generation",
    max_new_tokens=200,
    temperature=0.2,
    huggingfacehub_api_token=api_token
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("Hami: ", result.content)


print(chat_history)
