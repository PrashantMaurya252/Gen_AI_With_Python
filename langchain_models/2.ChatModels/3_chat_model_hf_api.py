# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv
# load_dotenv()

# llm = HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",task="text-generation")

# model = ChatHuggingFace(llm=llm)

# result = model.invoke("What is the capital of India ?")
# print(result)


from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ["HF_HOME"] = 'E:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    device=-1,  # CPU
    pipeline_kwargs={
        "max_new_tokens": 128,
        "do_sample": False,
        "temperature":1,
        "return_full_text": False,
    },
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of USA?")
print(result.content)