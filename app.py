import streamlit as st
import pandas as pd
import numpy as np
import hdbscan
import umap
import plotly.express as px

import json
import logging
import openai

from chromadb import PersistentClient
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_distances

from helper.cluster_naming import generate_cluster_name

# 0) Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)

load_dotenv()
logger.info("Environment variables loaded.")

embedding_model = "text-embedding-3-small"


def get_embedding(text: str):
    """
    Uses openai.Client() with model=embedding_model to create embeddings.
    """
    text = text.replace("\n", " ")
    embedding = openai.Client().embeddings.create(input=[text], model=embedding_model).data[0].embedding
    return embedding


#############################
# Query the DB
#############################

def query_chroma_db(
    query_text: str,
    distance_threshold: float = 0.75,
    max_n: int = 1000
) -> pd.DataFrame:
    """
    1) Embed 'query_text' with the chosen model.
    2) Query up to 'max_n' results from the collection.
    3) Filter out rows with distance > distance_threshold.
    4) Return a DataFrame of the filtered results.

    Note: If you use cosine distance, the distance is in [0,2].
          Typically, 'distance' ~ (1 - similarity), so
          distance_threshold=0.25 corresponds to similarity >= 0.75.
    """
    collection_name = "my_collection_1536"
    persist_path = r"S:\SID\Analytics\Working Files\Individual\Florian\Projects\semantic_search\Data\ChromaDB"

    logger.info(f"Embedding query text with model '{embedding_model}': {query_text[:60]}...")
    query_vector = get_embedding(query_text)

    logger.info("Connecting to Chroma DB (PersistentClient).")
    chroma_client = PersistentClient(path=persist_path)

    logger.info(f"Retrieving collection '{collection_name}'.")
    collection = chroma_client.get_collection(name=collection_name)

    logger.info(f"Querying up to {max_n} results from '{collection_name}'...")
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=max_n,
        include=["distances", "documents", "metadatas"]
    )

    ids = results["ids"][0]
    distances = results["distances"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    logger.info(f"Chroma returned {len(ids)} results before filtering.")

    # Build the DataFrame
    df_out = pd.DataFrame({"id": ids, "distance": distances, "document": documents})
    df_meta = pd.json_normalize(metadatas)
    df_final = pd.concat([df_out, df_meta], axis=1)

    # If 'embedding_long' is in columns, drop it
    if "embedding_long" in df_final.columns:
        df_final.drop(columns=["embedding_long"], inplace=True)

    # Parse 'embedding_short' from JSON, rename to 'embedding'
    if "embedding_short" in df_final.columns:
        df_final["embedding_short"] = df_final["embedding_short"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        df_final.rename(columns={"embedding_short": "embedding"}, inplace=True)

    # Apply the distance threshold filter
    pre_filter_count = len(df_final)
    df_final = df_final[df_final["distance"] <= distance_threshold]
    post_filter_count = len(df_final)

    logger.info(
        f"Applied distance threshold of {distance_threshold}. "
        f"Retained {post_filter_count} out of {pre_filter_count} results."
    )

    return df_final



# def query_chroma_db(query_text: str, top_n: int = 200) -> pd.DataFrame:
#     collection_name = "my_collection_1536"
#     persist_path = r"S:\SID\Analytics\Working Files\Individual\Florian\Projects\semantic_search\Data\ChromaDB"
#
#     logger.info(f"Embedding query text with model '{embedding_model}': {query_text[:60]}...")
#     query_vector = get_embedding(query_text)
#
#     # Persistent client
#     logger.info("Connecting to Chroma DB.")
#     chroma_client = PersistentClient(path=persist_path)
#
#     logger.info(f"Retrieving collection '{collection_name}'.")
#     collection = chroma_client.get_collection(name=collection_name)
#
#     logger.info(f"Querying top {top_n} results from '{collection_name}'.")
#     results = collection.query(
#         query_embeddings=[query_vector],
#         n_results=top_n,
#         include=["distances", "documents", "metadatas"]
#     )
#
#     ids = results["ids"][0]
#     distances = results["distances"][0]
#     documents = results["documents"][0]
#     metadatas = results["metadatas"][0]
#
#     logger.info(f"Received {len(ids)} results.")
#
#     # Build df
#     df_out = pd.DataFrame({"id": ids, "distance": distances, "document": documents})
#     df_meta = pd.json_normalize(metadatas)
#     df_final = pd.concat([df_out, df_meta], axis=1)
#
#     # If 'embedding_long' is in columns, drop it
#     if "embedding_long" in df_final.columns:
#         df_final.drop(columns=["embedding_long"], inplace=True)
#
#     # Parse 'embedding_short' from JSON, rename to 'embedding'
#     if "embedding_short" in df_final.columns:
#         df_final["embedding_short"] = df_final["embedding_short"].apply(
#             lambda x: json.loads(x) if isinstance(x, str) else x
#         )
#         df_final.rename(columns={"embedding_short": "embedding"}, inplace=True)
#
#     return df_final


##############################
# Caching / Data Processing
##############################

@st.cache_data
def cluster_and_reduce(df: pd.DataFrame,
                       min_cluster_size: int,
                       min_samples: int,
                       cluster_selection_epsilon: float) -> pd.DataFrame:
    df = df[df['embedding'].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    if len(df) == 0:
        return df

    mat = np.array(df['embedding'].tolist())
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon
    )
    labels = clusterer.fit_predict(mat)
    df["hdbscan_id"] = labels

    reducer = umap.UMAP(n_components=2)
    coords_2d = reducer.fit_transform(mat)
    df["x"] = coords_2d[:, 0]
    df["y"] = coords_2d[:, 1]

    return df


###########################################
#  CLUSTER NAMING LOGIC
###########################################

def name_clusters(df: pd.DataFrame,
                  cluster_col: str = "hdbscan_id",
                  embedding_col: str = "embedding",
                  text_col: str = "document",
                  top_k: int = 10,
                  skip_noise_label: int = -1) -> pd.DataFrame:
    """
    1) For each unique cluster ID in `cluster_col`, find the centroid of embeddings.
    2) Get the top_k closest items to that centroid.
    3) Use them to generate or guess a cluster name (placeholder).
    4) Create new column f"{cluster_col}_name".
    """
    df_out = df.copy()

    # Unique cluster IDs
    unique_ids = df_out[cluster_col].unique()
    cluster_id_to_name = {}

    for c_id in unique_ids:
        if skip_noise_label is not None and c_id == skip_noise_label:
            continue

        cluster_data = df_out[df_out[cluster_col] == c_id]
        if cluster_data.empty:
            continue

        # compute centroid
        embeddings = np.array(cluster_data[embedding_col].tolist())
        centroid = embeddings.mean(axis=0, keepdims=True)

        # find top_k closest
        dists = cosine_distances(centroid, embeddings).flatten()
        top_indices = np.argsort(dists)[:top_k]
        representative_texts = cluster_data.iloc[top_indices][text_col].tolist()

        cluster_name = generate_cluster_name(representative_texts)
        cluster_id_to_name[c_id] = cluster_name

    # Assign a "noise" name if needed
    name_col = f"{cluster_col}_name"
    df_out[name_col] = df_out[cluster_col].apply(lambda cid: cluster_id_to_name.get(cid, "Noise"))

    return df_out


###################
# 3) Streamlit App
###################

def main():
    st.title("Interactive Semantic Search + Clustering Demo")

    # A) Query input
    query_text = st.text_input("Enter your query:", value="The developers do a great job with the updates")
    distance_threshold = st.slider("Distance Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
    max_n: int = 1000

    # B) HDBSCAN parameters
    st.subheader("Clustering Parameters")
    min_cluster_size = st.number_input("min_cluster_size", value=10, min_value=2, max_value=100)
    min_samples = st.number_input("min_samples", value=2, min_value=1, max_value=100)
    cluster_selection_epsilon = st.slider("cluster_selection_epsilon", min_value=0.0, max_value=1.0, value=0.15,
                                          step=0.01)

    # If "Run Query" is clicked:
    if st.button("Run Query"):
        st.write("Querying the Vector Database...")

        df_results = query_chroma_db(query_text, distance_threshold=distance_threshold, max_n=max_n)
        st.write(f"Retrieved {len(df_results)} items from DB.")

        if len(df_results) == 0:
            st.warning("No results returned. Check your query or DB.")
            return

        # cluster + reduce
        st.write("Clustering + dimensionality reduction...")
        df_clustered = cluster_and_reduce(
            df_results,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon
        )
        st.write(f"After filtering, we have {len(df_clustered)} rows with valid embeddings.")

        # Save to session_state so we can name clusters later
        st.session_state["df_clustered"] = df_clustered
        # Also store the unique cluster IDs
        unique_cluster_ids = df_clustered["hdbscan_id"].unique()
        st.session_state["unique_cluster_ids"] = unique_cluster_ids

        st.write(f"Found {len(unique_cluster_ids)} distinct cluster IDs (including noise).")

        # Visualization
        if len(df_clustered) > 0 and "x" in df_clustered.columns and "y" in df_clustered.columns:
            fig = px.scatter(
                df_clustered,
                x="x", y="y",
                color="hdbscan_id",
                hover_data=["hdbscan_id", "document", "topic"],
                title="2D Projection (Colored by hdbscan_id)",
                width=800, height=600
            )
            st.plotly_chart(fig)

    # If we *already* have a df_clustered in session_state, allow naming
    if "df_clustered" in st.session_state and st.session_state["df_clustered"] is not None:
        df_clustered = st.session_state["df_clustered"]
        unique_ids = st.session_state["unique_cluster_ids"] if "unique_cluster_ids" in st.session_state else []

        # Show how many cluster IDs we have
        st.write(f"Found {len(unique_ids)} unique clusters.")

        # A separate button to name the clusters
        if st.button("Name Clusters"):
            st.write("Naming clusters... (placeholder logic)")
            df_named = name_clusters(
                df_clustered,
                cluster_col="hdbscan_id",
                embedding_col="embedding",
                text_col="document",
                top_k=10,
                skip_noise_label=-1
            )
            st.session_state["df_clustered"] = df_named  # store updated version

            # Show some info
            named_col = "hdbscan_id_name"
            if named_col in df_named.columns:

                # Re-visualize with new color
                fig2 = px.scatter(
                    df_named,
                    x="x", y="y",
                    color="hdbscan_id_name",
                    hover_data=["hdbscan_id_name", "document", "topic"],
                    title="2D Projection (Colored by Named Cluster)",
                    width=800, height=600
                )
                st.plotly_chart(fig2)

                st.write("Here's a sample of the DataFrame with cluster names:")
                st.dataframe(df_named.head(20))

        else:
            st.info("Click 'Name Clusters' to generate cluster names.")
    else:
        st.info("No data in memory yet. Please run a query first.")

if __name__ == "__main__":
    main()
