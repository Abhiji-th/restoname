from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel
from typing import Literal
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=800
)

model = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field("Sentiment of the following feedback") 

parser = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text as either positive or negative. \n {feedback}. \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="Generate a response for the following positive feedback. \n {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Generate a response for the following negative feedback. \n {feedback}",
    input_variables=["feedback"]
)

sentiment_chain = prompt1 | model | parser

str_parser = StrOutputParser()

response_chain = RunnableBranch(
    (lambda x: x.sentiment=="positive", prompt2 | model | str_parser),
    (lambda x: x.sentiment=="negative", prompt3 | model | str_parser),
    RunnableLambda(lambda x: "Sentiment not found")
)

chain = sentiment_chain | response_chain

result = chain.invoke({"feedback": "This is a terrible smartphone"})

print(result)