from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.6,
    max_new_tokens=100
)

model = ChatHuggingFace(llm=llm)

chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if(user_input=="exit"): 
        break
    result = model.invoke(chat_history)
    chat_history.append(result)
    print("\nBot:", result.content, "\n")
