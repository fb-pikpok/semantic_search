import os
import json
import pandas as pd
import numpy as np
import logging
import openai

import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
from dotenv import load_dotenv

# 0) Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)  # Suppress API HTTP request logs

######################
# 0) Environment Setup
######################

load_dotenv()  # Loads environment variables from .env if present
logger.info("Environment variables loaded.")

embedding_model = "text-embedding-3-small"


def get_embedding(text: str):
    """
    Your unmodified embedding function.
    Uses openai.Client() with model=embedding_model to create embeddings.
    """
    text = text.replace("\n", " ")
    embedding = openai.Client().embeddings.create(input=[text], model=embedding_model).data[0].embedding
    logger.info(f"Embedding: {embedding}")
    return embedding


###############################
# 2) Load & Prepare CSV / PKL
###############################

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load your DataFrame from a .pkl (or CSV, as you have in your script).
    Expects columns:
      - 'id'
      - 'embedding_long' (1536D)
      - 'embedding_short' (25D)
      - 'sentence' (optional)
      - plus other metadata...
    """
    logger.info(f"Loading DataFrame from: {csv_path}")
    df = pd.read_pickle(csv_path)  # or read_csv if that's actually what you have
    logger.info(f"DataFrame loaded with {len(df)} rows.")
    return df


##################################
# 3) Create & Populate Chroma DB
##################################

def create_and_populate_chroma(df: pd.DataFrame, collection_name: str, persist_path: str = "chroma_data") -> None:
    """
    Creates (or overwrites) a local persistent Chroma collection using `embedding_long` (1536D) as vectors.
    `embedding_short` is stored as JSON in the metadata to avoid Chroma's scalar-only metadata restriction.
    """


    logger.info("Initializing persistent Chroma client via `PersistentClient`.")
    chroma_client = chromadb.PersistentClient(path=persist_path)

    # If a collection with the same name exists, delete it
    existing = chroma_client.list_collections()  # returns a list of collection names in v0.6.0
    if collection_name in existing:
        logger.info(f"Collection '{collection_name}' already exists. Deleting it.")
        chroma_client.delete_collection(name=collection_name)

    logger.info(f"Creating new collection '{collection_name}'.")
    collection = chroma_client.create_collection(name=collection_name)


    # We'll parse the "embedding_long" column as lists (it might be JSON strings).
    def ensure_list(val):
        if isinstance(val, str):
            return json.loads(val)  # e.g. "[0.123, 0.456, ...]" -> Python list
        return val

    logger.info("Converting embedding_long values to Python lists.")
    df["embedding_long"] = df["embedding_long"].apply(ensure_list)

    # Prepare data for insertion
    all_ids = df["id"].astype(str).tolist()
    all_embeddings = df["embedding_long"].tolist()

    # If you want to store a text column as 'documents', e.g. "sentence":
    documents = df["sentence"].fillna("").astype(str).tolist() if "sentence" in df.columns else [""] * len(df)

    # Build metadata. We'll exclude 'embedding_long' to avoid big vectors in metadata.
    exclude_cols = {"id", "embedding_long"}
    meta_columns = [col for col in df.columns if col not in exclude_cols]

    metadatas = []
    for _, row in df.iterrows():
        meta = {}
        for col in meta_columns:
            val = row[col]
            # If it's a list (like embedding_short), convert to JSON string
            if isinstance(val, list):
                val = json.dumps(val)
            meta[col] = val
        metadatas.append(meta)

    # Insert in batches
    batch_size = 100
    num_rows = len(df)
    logger.info(f"Inserting {num_rows} rows into '{collection_name}' in batches of {batch_size}.")

    for start_idx in range(0, num_rows, batch_size):
        end_idx = start_idx + batch_size
        sub_ids = all_ids[start_idx:end_idx]
        sub_embeddings = all_embeddings[start_idx:end_idx]
        sub_docs = documents[start_idx:end_idx]
        sub_metas = metadatas[start_idx:end_idx]

        collection.add(
            ids=sub_ids,
            embeddings=sub_embeddings,
            documents=sub_docs,
            metadatas=sub_metas
        )
        logger.info(f"Inserted rows {start_idx} to {end_idx - 1} into the collection.")

    logger.info(f"Collection '{collection_name}' successfully created and populated at '{persist_path}'. "
                "You can reuse it in future runs without re-inserting.")


##################################
# 4) Query Function
##################################

def query_chroma(query_text: str, collection_name: str, top_n: int = 200, persist_path: str = "chroma_data") -> pd.DataFrame:
    """
    1) Generate a 1536D embedding for 'query_text' using OpenAI.
    2) Query the persistent Chroma DB for top_n results.
    3) Return a DataFrame WITHOUT the large embeddings,
       but keep 'embedding_short' in metadata (JSON string). We'll parse it
       and rename it to 'embedding' to avoid confusion.
    """
    logger.info(f"Embedding query text with model '{embedding_model}': {query_text[:60]}...")
    query_vector = get_embedding(query_text)

    logger.info("Connecting to the persistent Chroma DB for querying.")
    # Load from the same `persist_path` to see the existing data
    chroma_client = chromadb.PersistentClient(path=persist_path)

    logger.info(f"Retrieving collection '{collection_name}'.")
    collection = chroma_client.get_collection(name=collection_name)

    logger.info(f"Querying top {top_n} results from '{collection_name}'.")
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_n,
        include=["distances", "documents", "metadatas"]
    )

    # For a single query, results fields are list-of-lists
    ids = results["ids"][0]
    distances = results["distances"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    logger.info(f"Received {len(ids)} results.")

    # Build a DataFrame
    df_out = pd.DataFrame({"id": ids, "distance": distances, "document": documents})

    # Expand metadata
    df_meta = pd.json_normalize(metadatas)
    df_final = pd.concat([df_out, df_meta], axis=1)

    # If for some reason 'embedding_long' got included (it shouldn't), drop it
    if "embedding_long" in df_final.columns:
        df_final.drop(columns=["embedding_long"], inplace=True)

    # Parse 'embedding_short' from JSON, rename to 'embedding'
    if "embedding_short" in df_final.columns:
        df_final["embedding_short"] = df_final["embedding_short"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        df_final.rename(columns={"embedding_short": "embedding"}, inplace=True)

    return df_final


######################
# 5) Main / Example
######################

if __name__ == "__main__":
    # Adjust path and collection name as needed
    csv_path = r"S:\SID\Analytics\Working Files\Individual\Florian\Projects\semantic_search\Data\db_embedded_prepared.csv"
    collection_name = "my_collection_1536"
    persist_dir = "S:\SID\Analytics\Working Files\Individual\Florian\Projects\semantic_search\Data\ChromaDB"  # folder name for storing the persistent DB

    logger.info("Starting script...")

    # # 1) Load your DataFrame
    # df = load_data(csv_path)
    #
    # # 2) Populate the persistent Chroma DB with the 1536D embeddings
    # create_and_populate_chroma(df, collection_name, persist_path=persist_dir)

    # 3) Query the database with some text
    query_text = "People are happy with the gameplay experience"
    results_df = query_chroma(query_text, collection_name, top_n=200, persist_path=persist_dir)

    logger.info("Query returned a DataFrame with %d rows.", len(results_df))
    logger.info("Head of the results:\n%s", results_df.head(10).to_string())

    # Possibly save the results
    results_df.to_pickle("query_results.pkl")

    logger.info("Done.")
