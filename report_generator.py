from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.6
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write a detailed report about {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Summarise the following report in 5 lines. \n {text}",
    input_variables=["text"]
)

prompt1 = template1.format(topic="black hole")

result = model.invoke(prompt1)

prompt2 = template2.format(text=result.content)

final_result = model.invoke(prompt2)

print(final_result.content)


