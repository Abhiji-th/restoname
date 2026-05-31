from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    max_new_tokens=5000
)

model = ChatHuggingFace(llm=llm)

template = load_prompt("template.json")

st.header("Trip Planner")

trip_type = st.selectbox("Select trip type", ["Adventure", "Honeymoon", "Chill"])
destination = st.selectbox("Select destination", ["Wayanad", "Ooty", "Mysore"])
days = st.slider("Select days", min_value=1, max_value=10, value=2)

if st.button("Generate Itenary"):
    prompt = template.invoke({
        "trip_type": trip_type,
        "destination": destination,
        "days": days
    })

    result = model.invoke(prompt)
    print(len(result.content))

    st.write(result.content)
