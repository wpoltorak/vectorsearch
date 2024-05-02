import os
import pymongo
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = pymongo.MongoClient(os.environ.get("mongodb.connect_string"))
db = client.sample_mflix
collection = db.movies

openaiclient = OpenAI(api_key=os.environ.get("openai.api_key"))


def generate_embedding_hf(text: str) -> list[float]:
    response = requests.post(
        os.environ.get("hf.embedding_url"),
        headers={"Authorization": f"Bearer {os.environ.get('hf.token')}"},
        json={"inputs": text},
    )

    if response.status_code != 200:
        raise ValueError(
            f"Request failed with status code {response.status_code}: {response.text}"
        )

    return response.json()


def generate_embedding_openai(text: str) -> list[float]:
    response = openaiclient.embeddings.create(
        model=os.environ.get("openai.embedding"), input=text
    )

    return response.data[0].embedding


def printEmbedding(text: str):
    print(generate_embedding_hf(str))


for doc in collection.find({"plot": {"$exists": True}}).limit(50):
    doc["plot_embedding_hf"] = generate_embedding_hf(doc["plot"])
    collection.replace_one({"_id": doc["_id"]}, doc)
