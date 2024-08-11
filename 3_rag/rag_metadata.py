from dotenv import load_dotenv

import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

cur_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(cur_dir, "books")
persistent_dir = os.path.join(cur_dir, "db", "chroma_db_with_metadata")

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


if not os.path.exists(persistent_dir):
    print("DB does not exist. Initializing vector store")

    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"The file {books_dir} does not exist.")


    files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    documents= []
    for file in files:
        file_path = os.path.join(books_dir, file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": file}
            documents.append(doc)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunk Info. ---")
    print(f"Number of chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}")


    #Create Embeddings



    print("\n--- Finished creating embeddings ---")


    print("\n--- Creating Vector Store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_dir)
    print("\n-- Finished creating Vector store ---")

else:
    print("Vector store already exists. No need to initialize")

db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)
query = "How did Juliet die?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4},
)

relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


    



