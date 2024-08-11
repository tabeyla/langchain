from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

classn_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an helpful assistant"),
        ("human", "Given the feedback {feedback}, classify it as positive, negative, neutal, esclate")

    ]
)

pos_feedback = []