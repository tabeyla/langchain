from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

messages = [ 
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9")
    
]

result = model.invoke(messages)
print(result.content)

messages = [ 
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9"),
    AIMessage(content="81 divided by 9 is **9**"),
    HumanMessage(content="What is 10 times 5?")
]
result = model.invoke(messages)
print(result.content)

    

