from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader
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

url = "https://scikit-learn.org/stable/modules/lda_qda.html"

loader = WebBaseLoader(url)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Answer the question : {question}, based on the following text: \n {text}",
    input_variables=["question","text"]
)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({"question": "what is dimensionality reduction", "text": docs[0].page_content})

print(result)
