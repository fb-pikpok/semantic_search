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

# -----------------------------------------------------------------------------
# 0) Setup Streamlit Page + Logging
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide")

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


def query_chroma_db(
        query_text: str,
        distance_threshold: float = 0.75,
        max_n: int = 1000
) -> pd.DataFrame:
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

    # Build DataFrame
    df_out = pd.DataFrame({"id": ids, "distance": distances, "document": documents})
    df_meta = pd.json_normalize(metadatas)
    df_final = pd.concat([df_out, df_meta], axis=1)

    # Drop or rename embedding columns as needed
    if "embedding_long" in df_final.columns:
        df_final.drop(columns=["embedding_long"], inplace=True)

    if "embedding_short" in df_final.columns:
        df_final["embedding_short"] = df_final["embedding_short"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        df_final.rename(columns={"embedding_short": "embedding"}, inplace=True)

    # Apply threshold
    pre_filter_count = len(df_final)
    df_final = df_final[df_final["distance"] <= distance_threshold]
    post_filter_count = len(df_final)

    logger.info(
        f"Applied distance threshold of {distance_threshold}. "
        f"Retained {post_filter_count} out of {pre_filter_count} results."
    )

    return df_final


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


def name_clusters(df: pd.DataFrame,
                  cluster_col: str = "hdbscan_id",
                  embedding_col: str = "embedding",
                  text_col: str = "document",
                  top_k: int = 10,
                  skip_noise_label: int = -1) -> pd.DataFrame:
    df_out = df.copy()
    unique_ids = df_out[cluster_col].unique()
    cluster_id_to_name = {}

    for c_id in unique_ids:
        if skip_noise_label is not None and c_id == skip_noise_label:
            continue

        cluster_data = df_out[df_out[cluster_col] == c_id]
        if cluster_data.empty:
            continue

        embeddings = np.array(cluster_data[embedding_col].tolist())
        centroid = embeddings.mean(axis=0, keepdims=True)

        dists = cosine_distances(centroid, embeddings).flatten()
        top_indices = np.argsort(dists)[:top_k]
        representative_texts = cluster_data.iloc[top_indices][text_col].tolist()

        cluster_name = generate_cluster_name(representative_texts)
        cluster_id_to_name[c_id] = cluster_name

    name_col = f"{cluster_col}_name"
    df_out[name_col] = df_out[cluster_col].apply(lambda cid: cluster_id_to_name.get(cid, "Noise"))
    return df_out


def main():
    # Track steps as booleans in session state
    if "query_done" not in st.session_state:
        st.session_state["query_done"] = False
    if "cluster_done" not in st.session_state:
        st.session_state["cluster_done"] = False
    if "name_done" not in st.session_state:
        st.session_state["name_done"] = False

    st.title("Interactive Semantic Search + Clustering Demo")

    # -----------------------------
    # SIDEBAR: User Inputs
    # -----------------------------
    with st.sidebar:
        st.header("Search & Filter")
        query_text = st.text_input("Query:", value="The developers do a great job with the updates")
        distance_threshold = st.slider("Distance Threshold", 0.0, 1.0, 0.75, 0.01)
        max_n = 1000

        run_query_btn = st.button("Run Query")

        st.header("Clustering Settings")
        min_cluster_size = st.number_input("min_cluster_size", value=10, min_value=2, max_value=100)
        min_samples = st.number_input("min_samples", value=2, min_value=1, max_value=100)
        cluster_selection_epsilon = st.slider("cluster_selection_epsilon",
                                              min_value=0.0, max_value=1.0,
                                              value=0.15, step=0.01)

        cluster_data_btn = st.button("Cluster Data")
        name_clusters_btn = st.button("Name Clusters")

    # -----------------------------
    # 1) Run Query
    # -----------------------------
    if run_query_btn:
        with st.spinner("Running query..."):
            results_df = query_chroma_db(query_text, distance_threshold=distance_threshold, max_n=max_n)
        st.session_state["results_df"] = results_df

        st.session_state["query_done"] = True
        st.session_state["cluster_done"] = False
        st.session_state["name_done"] = False

        # Sort results
        sorted_results = results_df.sort_values(by="distance", ascending=True)
        st.session_state["sorted_results"] = sorted_results

        # Headline
        st.session_state["query_headline"] = (
            f"Query **'{query_text}'** returned **{len(results_df)}** results "
            f"above threshold = **{distance_threshold}**"
        )

        st.success(f"Done! Found {len(results_df)} results.")

    # Headline, if exists
    if st.session_state.get("query_headline", None):
        st.subheader(st.session_state["query_headline"])

    # Display head/tail if we have data
    if "results_df" in st.session_state and len(st.session_state["results_df"]) > 0:
        sorted_results = st.session_state["sorted_results"]
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Closest (Head)")
            st.dataframe(sorted_results[["distance", "document"]].head())
        with col2:
            st.write("#### Farthest (Tail)")
            st.dataframe(sorted_results[["distance", "document"]].tail())
    else:
        st.stop()

    # -----------------------------
    # 2) Cluster Data
    # -----------------------------
    if cluster_data_btn:
        if not st.session_state["query_done"] or "results_df" not in st.session_state:
            st.warning("No query results available. Please run a query first.")
        else:
            df_results = st.session_state["results_df"]
            if len(df_results) == 0:
                st.warning("No data to cluster. Please run a query that returns some rows.")
            else:
                with st.spinner("Clustering data..."):
                    df_clustered = cluster_and_reduce(
                        df_results,
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_epsilon=cluster_selection_epsilon
                    )
                st.session_state["df_clustered"] = df_clustered
                st.session_state["cluster_done"] = True
                st.session_state["name_done"] = False

                st.success(
                    f"Clustering complete. Found {df_clustered['hdbscan_id'].nunique()} clusters (including noise).")

                if len(df_clustered) > 0 and "x" in df_clustered.columns and "y" in df_clustered.columns:
                    fig = px.scatter(
                        df_clustered,
                        x="x", y="y",
                        color="hdbscan_id",
                        hover_data=["hdbscan_id", "document"]
                        if "topic" not in df_clustered.columns
                        else ["hdbscan_id", "document", "topic"],
                        title="2D Projection (Colored by hdbscan_id)",
                        width=800, height=600
                    )
                    st.plotly_chart(fig)

    # If no cluster data, stop
    if "df_clustered" not in st.session_state or len(st.session_state["df_clustered"]) == 0:
        st.stop()

    # -----------------------------
    # 3) Name Clusters
    # -----------------------------
    if name_clusters_btn:
        if not st.session_state["cluster_done"]:
            st.warning("Please cluster the data first before naming clusters.")
        else:
            with st.spinner("Naming clusters..."):
                df_named = name_clusters(
                    st.session_state["df_clustered"],
                    cluster_col="hdbscan_id",
                    embedding_col="embedding",
                    text_col="document",
                    top_k=10,
                    skip_noise_label=-1
                )
            st.session_state["df_clustered"] = df_named
            st.session_state["name_done"] = True

            fig2 = px.scatter(
                df_named,
                x="x", y="y",
                color="hdbscan_id_name",
                hover_data=["hdbscan_id_name", "document"]
                if "topic" not in df_named.columns
                else ["hdbscan_id_name", "document", "topic"],
                title="2D Projection (Colored by Named Cluster)",
                width=800, height=600
            )
            st.plotly_chart(fig2)

            st.write("Sample of the DataFrame with cluster names:")
            st.dataframe(df_named.head(20))

    # -----------------------------
    # 4) Download JSON
    # -----------------------------
    # Show this only if all steps done:
    if (st.session_state["query_done"]
            and st.session_state["cluster_done"]
            and st.session_state["name_done"]):
        st.info("All steps done! You can now download the final DataFrame below:")

        # Copy or rename columns for the JSON
        final_df = st.session_state["df_clustered"].copy()

        # Rename the coordinate columns only in the final df
        final_df.rename(columns={
            "x": "hdbscan_UMAP_2D_x",
            "y": "hdbscan_UMAP_2D_y"
        }, inplace=True)

        # Convert to JSON with the renamed columns
        json_str = final_df.to_json(orient="records")

        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="clustered_data.json",
            mime="application/json",
            key="download_json"
        )


if __name__ == "__main__":
    main()
