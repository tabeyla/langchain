from dotenv import load_dotenv

import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
load_dotenv()

hf_embeddings = HuggingFaceEmbeddings( model_name="all-MiniLM-L6-v2")
goog_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

cur_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(cur_dir, "db")
file_path = os.path.join(cur_dir,"books","odyssey.txt")



loader = TextLoader(file_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

docs = text_splitter.split_documents(documents)

def query_vector_store(query, store_name, embedding_function):
    
    persistent_dir = os.path.join(cur_dir, "db", store_name)

    if not os.path.exists(persistent_dir):
        print(f"Vector Store {store_name} does not exist")
    else:
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(persist_directory=persistent_dir, embedding_function=embedding_function)
        
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1},
        )

        relevant_docs = retriever.invoke(query)
        print("\n--- Relevant Documents ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

def create_vector_store(docs, embeddings, store_name):

    persistent_dir = os.path.join(cur_dir, "db", store_name)

    if not os.path.exists(persistent_dir):

        print("DB does not exist. Initializing vector store")
        Chroma.from_documents(docs, embeddings, persist_directory=persistent_dir)
    else:
        print("Vector store already exists. No need to initialize")

    print("\n-- Finished creating Vector store ---")



# create_vector_store(docs, hf_embeddings, "chroma_db_hf")
create_vector_store(docs, goog_embeddings, "chroma_db_goog")

query = "Who is Odysseus' wife ?"
query_vector_store(query,"chroma_db_goog",goog_embeddings)


    



