from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda


load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")


messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes")

]



prompt_template = ChatPromptTemplate.from_messages(messages)

lower_case = RunnableLambda(lambda x: x.lower())

chain = prompt_template | model | StrOutputParser() | lower_case 

result = chain.invoke({"topic": "lawyers", "joke_count": 3})
print(result)