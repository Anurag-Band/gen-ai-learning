from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# loading pdf
pdf_path = Path(__file__).parent / "react-notes.pdf"

loader = PyPDFLoader(pdf_path)

docs = loader.load()

# split docs into chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
chunks = text_splitter.split_documents(documents=docs)

# embedding chunks into vector db

print("Starting the indexting docs into vector db")

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

vector_db = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

print("Indexing of documents done....")

