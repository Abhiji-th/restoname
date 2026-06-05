from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
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

loader = DirectoryLoader(
    path="PDFs",
    loader_cls=PyPDFLoader,
    glob="*.pdf"
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Generate a quiz on maps using the following text. \n {text}",
    input_variables=["text"]
)

docs = loader.lazy_load()

for doc in docs:
    print(doc.metadata)

