from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

query = "What is valorant"

docs =   retriever.invoke(query)

for i,doc in enumerate(docs):
    print(doc.page_content)