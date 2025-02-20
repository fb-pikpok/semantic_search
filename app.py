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
# 0) Initial Setup & Logging
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide")  # wide layout
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)

load_dotenv()
logger.info("Environment variables loaded.")

embedding_model = "text-embedding-3-small"


def get_embedding(text: str):
    text = text.replace("\n", " ")
    embedding = openai.Client().embeddings.create(input=[text], model=embedding_model).data[0].embedding
    return embedding


def query_chroma_db(query_text: str, distance_threshold: float = 0.75, max_n: int = 1000) -> pd.DataFrame:
    """
    1) Embed 'query_text'.
    2) Query up to 'max_n' results from the collection.
    3) Filter rows by distance <= distance_threshold.
    4) Return DataFrame.
    """
    collection_name = "my_collection_1536"
    persist_path = r"S:\SID\Analytics\Working Files\Individual\Florian\Projects\semantic_search\Data\ChromaDB"

    logger.info(f"Embedding query text with model '{embedding_model}': {query_text[:60]}...")
    query_vector = get_embedding(query_text)

    logger.info("Connecting to Chroma DB (PersistentClient).")
    chroma_client = PersistentClient(path=persist_path)

    logger.info(f"Retrieving collection '{collection_name}'.")
    collection = chroma_client.get_collection(name=collection_name)

    logger.info(f"Querying up to {max_n} results...")
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=max_n,
        include=["distances", "documents", "metadatas"]
    )

    ids = results["ids"][0]
    distances = results["distances"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    df_out = pd.DataFrame({"id": ids, "distance": distances, "document": documents})
    df_meta = pd.json_normalize(metadatas)
    df_final = pd.concat([df_out, df_meta], axis=1)

    # Clean up embedding columns if needed
    if "embedding_long" in df_final.columns:
        df_final.drop(columns=["embedding_long"], inplace=True)
    if "embedding_short" in df_final.columns:
        df_final["embedding_short"] = df_final["embedding_short"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        df_final.rename(columns={"embedding_short": "embedding"}, inplace=True)

    # Filter by threshold
    df_final = df_final[df_final["distance"] <= distance_threshold]

    return df_final


@st.cache_data
def cluster_and_reduce(df: pd.DataFrame,
                       min_cluster_size: int,
                       min_samples: int,
                       cluster_selection_epsilon: float) -> pd.DataFrame:
    """
    1) Filter out invalid embeddings
    2) Run HDBSCAN
    3) Run UMAP (2D) and store as x, y
    """
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
    """
    Find a centroid for each cluster, pick top_k nearest docs,
    and pass them to generate_cluster_name().
    """
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
    # Initialize session state flags for each step
    if "query_done" not in st.session_state:
        st.session_state["query_done"] = False
    if "cluster_done" not in st.session_state:
        st.session_state["cluster_done"] = False
    if "name_done" not in st.session_state:
        st.session_state["name_done"] = False

    st.title("Semantic search DEMO")


    # ------------------------------------------------------------------------------
    # STEP 1: Query
    # ------------------------------------------------------------------------------


    col1, col2 = st.columns([0.7, 0.3])

    with col1:
        st.markdown("#### What topic do you want to investigate?")
        # Query + threshold
        query_text = st.text_input("Enter your query:", value="The developers do a great job with the updates")
        distance_threshold = st.slider("Distance Threshold",
                                       min_value=0.0, max_value=1.0,
                                       value=0.75, step=0.01)

        # "Run Query" button
        if st.button("Run Query"):
            with st.spinner("Querying the database..."):
                results_df = query_chroma_db(query_text, distance_threshold=distance_threshold, max_n=1000)
            st.session_state["results_df"] = results_df

            # Mark step 1 as done
            st.session_state["query_done"] = True
            # Reset cluster steps
            st.session_state["cluster_done"] = False
            st.session_state["name_done"] = False

            # Create a persistent "headline"
            st.session_state["headline"] = (
                f"**{len(results_df)}** statements relate to your search."
            )

    with col2:
        st.markdown("#### Tips for Distance Threshold")
        st.markdown(''':blue-background[A threshold between **0.45 and 0.65** often works well. ]''')
        st.markdown(''':blue-background[If the tail results dont match your query, lower the threshold.]''')



    # Show the results (if any)
    if st.session_state["query_done"] and "results_df" in st.session_state:
        df = st.session_state["results_df"]
        if len(df) == 0:
            st.warning("No results. Try adjusting your query or threshold.")
        else:
            st.subheader(st.session_state["headline"])
            sorted_df = df.sort_values(by="distance", ascending=True)

            c1, c2, c3 = st.columns([0.35, 0.35, 0.3])
            with c1:
                st.write("**Closest (Head)**")
                st.dataframe(sorted_df[["distance", "document"]].head())
            with c2:
                st.write("**Farthest (Tail)**")
                st.dataframe(sorted_df[["distance", "document"]].tail())
            with c3:
                st.markdown("#### How to interpret the results?")
                st.markdown(''':blue-background[Left Table = Top 5 closest user statements **Head**]''')
                st.markdown(''':blue-background[Right Table = Top 5 farthest user statements **Tail**]''')
                st.markdown(''':blue-background[If the **Tail** results dont match your query, lower the threshold.]''')

    else:
        st.stop()  # Nothing to do yet, user hasn’t run the query

    # ------------------------------------------------------------------------------
    # STEP 2: Clustering
    # ------------------------------------------------------------------------------
    # Show clustering parameters only after we have some query results
    if st.session_state["query_done"] and len(st.session_state["results_df"]) > 0:
        st.markdown("---")
        st.markdown("#### Step 2: Cluster the Data")

        col3, col4 = st.columns([0.6, 0.4])
        with col3:
            min_cluster_size = st.number_input("min_cluster_size", value=10, min_value=2, max_value=100)
            min_samples = st.number_input("min_samples", value=2, min_value=1, max_value=100)
            cluster_selection_epsilon = st.slider("cluster_selection_epsilon",
                                                  min_value=0.0, max_value=1.0,
                                                  value=0.15, step=0.01)

            if st.button("Cluster Data"):
                df_query_results = st.session_state["results_df"]
                if len(df_query_results) == 0:
                    st.warning("No data to cluster. Please run a query first.")
                else:
                    with st.spinner("Clustering + dimensionality reduction..."):
                        df_clustered = cluster_and_reduce(
                            df_query_results,
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            cluster_selection_epsilon=cluster_selection_epsilon
                        )
                    st.session_state["df_clustered"] = df_clustered
                    st.session_state["cluster_done"] = True
                    st.session_state["name_done"] = False  # reset naming step

        with col4:
            st.markdown("#### Tips for Clustering Parameters")
            st.markdown(''':blue-background[**min_cluster_size** What is the smallest cluster you want to find?]''')
            st.markdown(''':blue-background[**min_samples** influences how dense a region must be to form a cluster.]''')
            st.markdown('''Essentially this measures how conservative the algorithm is in finding clusters. If you expect to have a lot of smaller topics paired with a few larger topics, you may want to increase this value.''')
            st.markdown(''':blue-background[**cluster_selection_epsilon** Typically you don’t want to touch this.]''')
            st.markdown('''This parameter can be useful if you want to adjust the granularity of the clusters. A smaller value will result in more clusters, while a larger value will result in fewer clusters.''')

    # Show cluster output if done
    if st.session_state["cluster_done"] and "df_clustered" in st.session_state:
        df_clustered = st.session_state["df_clustered"]
        st.session_state["headline2"] = (
            f"**{df_clustered['hdbscan_id'].nunique() -1}** Cluster(s) found."
        )

        st.subheader(st.session_state["headline2"])

        col5, col6 = st.columns([0.6, 0.4])
        with col5:
            if len(df_clustered) > 0 and "x" in df_clustered.columns and "y" in df_clustered.columns:
                fig = px.scatter(
                    df_clustered,
                    x="x", y="y",
                    color="hdbscan_id",
                    hover_data=["hdbscan_id", "document"]
                      if "topic" not in df_clustered.columns
                      else ["hdbscan_id", "document", "topic"],
                    #title="2D Projection (Colored by clusterID)",
                    width=1000, height=600
                )
                st.plotly_chart(fig)

        with col6:
            st.markdown("#### How to interpret the results?")
            st.markdown(''':blue-background[Each Datapoint is a user statement, hover over them to get more details.]''')
            st.markdown(''':blue-background[The colors represent the cluster ID assigned by the algorithm with your chosen parameters.]''')
            st.markdown(''':blue-background[Compare the shapes you see with the number of clusters. If individual shapes are colored differently, the clustering worked well.]''')
    else:
        st.stop()  # The user hasn’t clustered yet or no data to show

    # ------------------------------------------------------------------------------
    # STEP 3: Name Clusters
    # ------------------------------------------------------------------------------
    # Show naming step if we have cluster results
    if st.session_state["cluster_done"] and len(st.session_state["df_clustered"]) > 0:
        st.markdown("---")

        if st.button("Name Clusters"):
            df_clustered = st.session_state["df_clustered"]
            with st.spinner("Naming clusters..."):
                df_named = name_clusters(
                    df_clustered,
                    cluster_col="hdbscan_id",
                    embedding_col="embedding",
                    text_col="document",
                    top_k=10,
                    skip_noise_label=-1
                )
            st.session_state["df_clustered"] = df_named
            st.session_state["name_done"] = True

    if st.session_state["name_done"] and "df_clustered" in st.session_state:
        df_named = st.session_state["df_clustered"]
        st.success("Cluster naming complete!")
        col7, col8 = st.columns([0.6, 0.4])
        with col7:
    # Show results after naming
            fig2 = px.scatter(
                df_named,
                x="x", y="y",
                color="hdbscan_id_name",
                hover_data=["hdbscan_id_name", "document"]
                  if "topic" not in df_named.columns
                  else ["hdbscan_id_name", "document", "topic"],
                #title="2D Projection (Colored by Named Cluster)",
                width=1000, height=600
            )
            st.plotly_chart(fig2)

        with col8:
            st.markdown("#### How to interpret the results?")
            st.markdown(''':blue-background[ChatGPT was used to give the clusters a name so you know immediately what they are about.]''')
            st.markdown(''':blue-background[Click on **Noise** to hide outliers and focus on the main clusters.]''')

        # ------------------------------------------------------------------------------
        # FINAL: Download JSON
        # ------------------------------------------------------------------------------
        st.markdown("---")
        st.info("All steps done! You can now download the final DataFrame below:")

        # For the final export, rename x/y columns if needed, then export to JSON
        final_df = df_named.copy()
        # Example: rename columns only for JSON
        final_df.rename(columns={
            "x": "hdbscan_UMAP_2D_x",
            "y": "hdbscan_UMAP_2D_y"
        }, inplace=True)

        json_str = final_df.to_json(orient="records")
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="clustered_data.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
