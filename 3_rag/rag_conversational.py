from dotenv import load_dotenv

import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

cur_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cur_dir, "books", "langchain_demo.txt")
persistent_dir = os.path.join(cur_dir, "db", "chroma_db")

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


if not os.path.exists(persistent_dir):
    print("DB does not exist. Initializing vector store")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")


    loader = TextLoader(file_path)
    documents = loader.load()

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
query = "How can I learn more about langchain"

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

relevant_docs = retriever.invoke(query)

# print("\n--- Relevant Documents ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")
#     if doc.metadata:
#         print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


combined_input = (
    "Here are some documents that might help answer the questions: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on provided documents. If answer is not found in documents, respond with I'm not sure."

)

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

messages = [
    SystemMessage(content="You are helpful assistant"),
    HumanMessage(content=combined_input)

]

result = model.invoke(messages)

print(result)



