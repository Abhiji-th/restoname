from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=800
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Tell me a joke about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

def word_length_counter(x):
    return len(x.split())

word_length_runnable = RunnableLambda(word_length_counter)

joke_chain = prompt1 | model | parser

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": word_length_runnable
})

chain = joke_chain | parallel_chain

result = chain.invoke({"topic": "valorant"})

print(result)