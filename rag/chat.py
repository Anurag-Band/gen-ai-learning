from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

openai_client = OpenAI()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")


vector_db = QdrantVectorStore.from_existing_collection(
    collection_name="learning_rag",
    url="http://localhost:6333",
    embedding=embeddings_model,
)


user_query = input("Ask Something: 👉 ")

search_results = vector_db.similarity_search(user_query)

context = "\n\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"for result in search_results])


SYSTEM_PROMPT = f"""
    you are an intelligent Ai assistant who answers users query based on available context retrived from a PDF file along with page_content and page_number. If you don't have any context related to given query then just say, sorry bro, I don't found any references to you query in available docs.

    Context:
    {context}
"""

response = openai_client.chat.completions.create(
    model="gpt-5-mini",
    messages=[
        { "role": "system", "content": SYSTEM_PROMPT },
        { "role": "user", "content": user_query }
    ]
)

print(f"🤖: {response.choices[0].message.content}")