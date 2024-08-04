from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel


load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")


messages = [
    ("system", "You are an expert product reviwer"),
    ("human", "List the main features of the product {product_name}")

]


prompt_template = ChatPromptTemplate.from_messages(messages)


pros = [
    ("system", "You are an expert product reviwer"),
    ("human", "Given these features: {features}, list the pros of these features")

]

cons = [
    ("system", "You are an expert product reviwer"),
    ("human", "Given these features: {features}, list the cons of these features")

]

pros_branch_chain = (

    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()

)

cons_brach_chain = (

    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)


def combine_pros_cons(pros, cons):
    return f"\n------------------Pros:------------------\n{pros}\n\n------------------Cons:-------------------\n{cons}"


def analyze_pros(features):

    pros_template = ChatPromptTemplate.from_messages(pros)
    return pros_template.format_prompt(features=features)

def analyze_cons(features):

    cons_template = ChatPromptTemplate.from_messages(cons)
    return cons_template.format_prompt(features=features)
chain = (
    prompt_template 
    | model 
    | StrOutputParser() 
    | RunnableParallel(branches={"pros" : pros_branch_chain, "cons" :  cons_brach_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
    )

result = chain.invoke({"product_name": "Macbook Pro M1 2019"})

print(result)

    