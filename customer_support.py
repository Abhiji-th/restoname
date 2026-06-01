from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.6,
    max_new_tokens=50
)

model = ChatHuggingFace(llm=llm)

chat_template = ChatPromptTemplate([
    ("system", "You are an customer support agent for {domain}"),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{query}")
])

chat_history = []

with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())


prompt = chat_template.invoke({"domain": "electronics", "chat_history":chat_history, "query": "What is the status of my refund"})

print(prompt)