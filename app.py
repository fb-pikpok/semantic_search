import streamlit as st
import pandas as pd
import numpy as np
import hdbscan
import umap
import plotly.express as px


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

# We'll assume you already have these installed:
#   pip install streamlit hdbscan umap-learn plotly

#############################
# 1) Utility: Query Your DB
#############################

def query_chroma_db(query_text: str, top_n: int = 200) -> pd.DataFrame:
    """
    1) Generate a 1536D embedding for 'query_text' using OpenAI.
    2) Query the persistent Chroma DB for top_n results.
    3) Return a DataFrame WITHOUT the large embeddings,
       but keep 'embedding_short' in metadata (JSON string). We'll parse it
       and rename it to 'embedding' to avoid confusion.
    """
    collection_name = "my_collection_1536"
    persist_path = "S:\SID\Analytics\Working Files\Individual\Florian\Projects\semantic_search\Data\ChromaDB"


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


##############################
# 2) Caching / Data Processing
##############################

@st.cache_data
def cluster_and_reduce(df: pd.DataFrame,
                       min_cluster_size: int,
                       min_samples: int,
                       cluster_selection_epsilon: float) -> pd.DataFrame:
    """
    Given a DataFrame with 'embedding' (list of floats) for each row:
      1) Convert 'embedding' -> a NumPy matrix
      2) Cluster with HDBSCAN
      3) Reduce to 2D using UMAP
      4) Append 'hdbscan_id', 'x', 'y' columns to the DataFrame

    Returns a new DataFrame with the appended columns.
    """

    # Filter rows that have valid embeddings
    df = df[df['embedding'].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()

    if len(df) == 0:
        return df  # no embeddings, nothing to cluster

    # 1) Build matrix
    mat = np.array(df['embedding'].tolist())

    # 2) HDBSCAN
    hdbscan_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon
    )
    cluster_labels = hdbscan_clusterer.fit_predict(mat)
    df["hdbscan_id"] = cluster_labels

    # 3) UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    coords_2d = reducer.fit_transform(mat)
    df["x"] = coords_2d[:, 0]
    df["y"] = coords_2d[:, 1]

    return df


###################
# 3) Streamlit App
###################

def main():
    st.title("Interactive Semantic Search + Clustering Demo")

    # A) Query input
    query_text = st.text_input("Enter your query:", value="People are happy with the gameplay experience")
    top_n = st.number_input("Number of results (top_n):", value=200, min_value=1, max_value=2000)

    # B) HDBSCAN parameters
    st.subheader("Clustering Parameters")
    min_cluster_size = st.number_input("min_cluster_size", value=10, min_value=2, max_value=500)
    min_samples = st.number_input("min_samples", value=2, min_value=1, max_value=500)
    cluster_selection_epsilon = st.slider("cluster_selection_epsilon", min_value=0.0, max_value=1.0, value=0.15,
                                          step=0.01)

    # C) A button to run the query
    if st.button("Run Query"):
        st.write("Querying the Vector Database...")
        df_results = query_chroma_db(query_text, top_n=top_n)
        st.write(f"Retrieved {len(df_results)} items from DB.")

        # If no results, skip clustering
        if len(df_results) == 0:
            st.warning("No results returned from DB. Check your query or DB content.")
            return

        # D) Cluster & reduce
        st.write("Clustering and dimensionality reduction...")
        df_clustered = cluster_and_reduce(
            df_results,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon
        )

        st.write(f"After filtering, we have {len(df_clustered)} rows with valid embeddings for clustering.")

        # E) Visualization
        if len(df_clustered) > 0 and "x" in df_clustered.columns and "y" in df_clustered.columns:
            # Plotly scatter
            fig = px.scatter(
                df_clustered,
                x="x",
                y="y",
                color="hdbscan_id",
                hover_data=["id", "hdbscan_id"],
                title="2D Projection of the Query Results (Colored by Cluster)",
                width=800,
                height=600
            )
            st.plotly_chart(fig)
        else:
            st.info("Not enough data to visualize or missing embeddings.")

    else:
        st.info("Enter a query and click 'Run Query' to see the results.")


if __name__ == "__main__":
    main()
