# Step 1: Your source documents
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents = docs,
    embedding=embedding_model,
    collection_name="my_collection"
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 3, 'lambda_unit':0.5}    
)

query = "What is langchain"

result = retriever.invoke(query)

for doc in result:
    print(doc.page_content)
