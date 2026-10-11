from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-5.6-luna",
    reasoning_effort="none"
)

llm = model | StrOutputParser()

result = llm.invoke("What is the capital of India?")
print(result)  # Plain string