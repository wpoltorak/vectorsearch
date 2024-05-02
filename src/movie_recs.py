import os
import sys
import pymongo
from enum import Enum
from movie_embds import generate_embedding_hf
from movie_embds import generate_embedding_openai
from dotenv import load_dotenv

QUERY_CANDIDATES_NUMBER = 100
QUERY_RESULTS_NUMBER = 4
INDEX_NAME = "PlotSemanticSearch"


class EmbeddingEngine(Enum):
    HF = 1
    OPENAI = 2


def process_embedding_engine(engine) -> EmbeddingEngine:
    try:
        return EmbeddingEngine(engine)
    except ValueError:
        raise ValueError(f"{engine} is not a valid embedding engine")


load_dotenv()

client = pymongo.MongoClient(os.environ.get("db.connect_string"))
db = client.sample_mflix

query = "a movie about distant world in a galaxy far far away"

type = (
    process_embedding_engine(int(sys.argv[1]))
    if len(sys.argv) > 1
    else EmbeddingEngine.HF
)

match type:
    case EmbeddingEngine.HF:
        results = db.movies.aggregate(
            [
                {
                    "$vectorSearch": {
                        "queryVector": generate_embedding_hf(query),
                        "path": "plot_embedding_hf",
                        "numCandidates": QUERY_CANDIDATES_NUMBER,
                        "limit": QUERY_RESULTS_NUMBER,
                        "index": INDEX_NAME,
                    }
                }
            ]
        )
    case EmbeddingEngine.OPENAI:
        results = db.embedded_movies.aggregate(
            [
                {
                    "$vectorSearch": {
                        "queryVector": generate_embedding_openai(query),
                        "path": "plot_embedding",
                        "numCandidates": QUERY_CANDIDATES_NUMBER,
                        "limit": QUERY_RESULTS_NUMBER,
                        "index": INDEX_NAME,
                    }
                }
            ]
        )


for doc in results:
    print(f"Movie name: {doc['title']}, movie plot: {doc['plot']}\n")


def printItems(items):
    for item in items:
        print(item)
