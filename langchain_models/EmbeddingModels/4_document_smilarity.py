from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embedding=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

documents = [
    "Sachin Tendulkar is known as the Master Blaster and is one of the greatest batsmen in cricket history. He represented India for over two decades, scored 100 international centuries, and was part of India's 2011 World Cup-winning team.",

    "Virat Kohli is known for his consistent batting, aggressive approach, and exceptional ability to chase targets. He has captained India across all formats and played a major role in India's victories in the 2011 ODI World Cup and 2024 T20 World Cup.",

    "MS Dhoni is celebrated for his calm leadership, quick wicketkeeping, and ability to finish close matches. Under his captaincy, India won the 2007 T20 World Cup, the 2011 ODI World Cup, and the 2013 ICC Champions Trophy.",

    "Rohit Sharma is an opening batsman known for his elegant stroke play and powerful six-hitting. He has scored three double centuries in ODI cricket, including a record 264 runs, and captained India to victory in the 2024 T20 World Cup.",

    "Jasprit Bumrah is a fast bowler known for his unusual bowling action, accurate yorkers, and excellent control. His ability to take wickets and restrict runs across all formats has made him an essential member of India's bowling attack.",

    "Yuvraj Singh was an all-rounder known for his powerful left-handed batting and useful left-arm spin. He famously hit six sixes in one over during the 2007 T20 World Cup and was named Player of the Tournament at the 2011 ODI World Cup."
]

query = "Tell me about Yuvraj Singh"

doc_embeddings = embedding.embed_documents(documents)
query_embeddings = embedding.embed_query(query)

scores = cosine_similarity([query_embeddings],doc_embeddings)[0]
index, score=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(documents[index])
print("Similarity Score",score)