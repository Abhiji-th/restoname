from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = ["Reyna is a duelist",
             "Viper is a controller",
             "Sage is a sentinel",
             "Gekko is an initiator"]

query = "What kind of agent is Sage"

document_score = embedding.embed_documents(documents)
query_score = embedding.embed_query(query)

score = list(enumerate(cosine_similarity([query_score], document_score)[0]))
score = [(i, float(s)) for (i, s) in score]
index, score = sorted(score, key=lambda x:x[1])[-1]
print(score)

print(query)
print(documents[index])
print(f"Similarity Score:", score)