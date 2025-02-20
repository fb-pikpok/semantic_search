import json
import pandas as pd

import logging
import openai
import chromadb

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)


load_dotenv()  # Loads environment variables from .env if present
logger.info("Environment variables loaded.")

embedding_model = "text-embedding-3-small"


# region 0: Helper functions
# Create an embedding for the User query with OpenAi for the semantic search
def get_embedding(text: str):
    """
    Your unmodified embedding function.
    Uses openai.Client() with model=embedding_model to create embeddings.
    """
    text = text.replace("\n", " ")
    embedding = openai.Client().embeddings.create(
        input=[text],
        model=embedding_model
    ).data[0].embedding
    return embedding

# Prepare the db_embedded to be stored in the ChromaDB
def prepare_dataframe(
    data_source: str,
    input_json_path: str,
    output_csv_path: str = None
) -> pd.DataFrame:
    """
    1) Load JSON data into a Pandas DataFrame.
    2) Rename 'embedding' -> 'embedding'.
    3) Reduce embedding dimensions to 25 via UMAP, save as 'embedding_short'.
    4) Generate ID: 'steam_1', 'steam_2', ...
    5) (Optional) Write the resulting DataFrame to a CSV file.
    """

    # 1) Load JSON data
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # data is expected to be a list of dicts

    df = pd.DataFrame(data)


    # 2) Generate an incremental ID: steam_1, steam_2, ...
    df["pp_id"] = [f"{data_source}_{i+1}" for i in range(len(df))]

    # Optionally write to CSV
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)

    # Fill NaN values with empty strings
    df = df.fillna("")
    return df
#endregion


#region 1 Chroma interaction
def upsert_chroma_data(
        df: pd.DataFrame,
        collection_name: str,
        persist_path: str = "chroma_data",
        batch_size: int = 100
) -> None:
    """
    This function either creates the Chroma collection (if it doesn't exist)
    or retrieves it (if it does). It then:
      1. If the collection already existed, deletes rows matching the IDs in `df["pp_id"]`
      2. Adds (inserts) the new data in batches
    Ensuring NO duplicates.

    Args:
        df (pd.DataFrame): DataFrame containing columns:
            - "pp_id" (unique ID for each row)
            - "embedding" (list of floats or JSON string)
            - "sentence" (optional; stored as the 'documents' in Chroma)
            - plus other metadata columns
        collection_name (str): Name of the collection to create or retrieve.
        persist_path (str): Folder path for the persistent Chroma DB.
        batch_size (int): Number of rows to insert per batch (default=100).
    """

    # 1) Connect to (or create) the persistent Chroma client
    logger.info("Initializing persistent Chroma client via `PersistentClient`.")
    chroma_client = chromadb.PersistentClient(path=persist_path)

    existing_collections = chroma_client.list_collections()

    # 2) Either retrieve the existing collection OR create a new one
    newly_created = False
    if collection_name in existing_collections:
        logger.info(f"Collection '{collection_name}' exists. Retrieving it.")
        collection = chroma_client.get_collection(name=collection_name)
    else:
        logger.info(f"Collection '{collection_name}' does not exist. Creating a new one.")
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "hnsw:search_ef": 100}
        )
        newly_created = True

    # 3) Convert the "embedding" column to lists of floats (if stored as JSON strings)
    def ensure_list(val):
        if isinstance(val, str):
            return json.loads(val)  # e.g. "[0.123, 0.456]" -> Python list
        return val

    logger.info("Ensuring 'embedding' column is a list of floats.")
    df["embedding"] = df["embedding"].apply(ensure_list)

    # Prepare data
    all_ids = df["pp_id"].astype(str).tolist()
    all_embeddings = df["embedding"].tolist()

    # If there's a "sentence" column, store it as the 'documents'
    documents = (
        df["sentence"].fillna("").astype(str).tolist()
        if "sentence" in df.columns else [""] * len(df)
    )

    # Exclude 'pp_id' and 'embedding' from metadata
    exclude_cols = {"pp_id", "embedding"}
    meta_columns = [col for col in df.columns if col not in exclude_cols]

    # Convert each row of metadata to a dict
    metadatas = []
    for _, row in df.iterrows():
        meta = {}
        for col in meta_columns:
            val = row[col]
            if isinstance(val, list):
                val = json.dumps(val)  # store list-like data as a JSON string
            meta[col] = val
        metadatas.append(meta)

    # 4) Only if the collection already existed, delete any existing IDs first
    if not newly_created:
        logger.info(f"Deleting any existing items with these {len(all_ids)} IDs (collection was not newly created).")
        if all_ids:
            collection.delete(ids=all_ids)

    # 5) Insert new items in batches
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
        logger.info(f"Inserted rows {start_idx} to {end_idx - 1}.")

    logger.info(f"Upsert operation complete. Collection '{collection_name}' updated with no duplicates.")


# Function for querying the ChromaDB and returning the results
# extended with optional metadata filter (where clause)
def query_chroma(
    query_text: str,
    collection_name: str,
    similarity_threshold: float = 0.54,
    initial_top_n: int = 5500,
    persist_path: str = "chroma_data",
    where_filters: dict = None  # <--- new argument
) -> pd.DataFrame:
    """
    Query the ChromaDB collection to retrieve all results above a certain similarity threshold,
    optionally filtered by metadata.
    """

    query_vector = get_embedding(query_text)
    chroma_client = chromadb.PersistentClient(path=persist_path)
    logger.info(f"Retrieving collection '{collection_name}'.")
    collection = chroma_client.get_collection(name=collection_name)

    # Build query arguments
    query_args = {
        "query_embeddings": [query_vector],
        "n_results": initial_top_n,
        "include": ["distances", "documents", "metadatas"],
    }
    if where_filters:
        query_args["where"] = where_filters

    results = collection.query(**query_args)


    # For a single query, results fields are list-of-lists
    ids = results["ids"][0]
    distances = results["distances"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    logger.info(f"Received {len(ids)} results. Filtering based on similarity threshold.")

    # Filter results based on the similarity threshold
    filtered_data = [
        (id_, dist, doc, meta) for id_, dist, doc, meta in zip(ids, distances, documents, metadatas)
        if dist <= similarity_threshold
    ]

    if not filtered_data:
        #check if where filters are the reason for no results
        if where_filters:
            logger.info("No results found above the similarity threshold with the given metadata filters.")
        else:
            logger.info("No results found above the similarity threshold.")
        return pd.DataFrame()  # Return an empty DataFrame if no results match the threshold

    # Unzip the filtered data
    filtered_ids, filtered_distances, filtered_documents, filtered_metadatas = zip(*filtered_data)

    # Build a DataFrame
    df_out = pd.DataFrame({
        "pp_id": filtered_ids,
        "distance": filtered_distances,
        "document": filtered_documents
    })

    # Expand metadata
    df_meta = pd.json_normalize(filtered_metadatas)
    df_final = pd.concat([df_out, df_meta], axis=1)

    # Drop 'embedding' if it exists
    if "embedding" in df_final.columns:
        df_final.drop(columns=["embedding"], inplace=True)

    # Parse 'embedding_short' from JSON, rename to 'embedding'
    if "embedding_short" in df_final.columns:
        df_final["embedding_short"] = df_final["embedding_short"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        df_final.rename(columns={"embedding_short": "embedding"}, inplace=True)

    logger.info(f"Filtered results count: {len(df_final)}")
    return df_final

# endregion


if __name__ == "__main__":
    # Adjust path and collection name as needed
    input_path = r'S:\SID\Analytics\Working Files\Individual\Florian\Projects\semantic_search\Database\Backup\Google Play\db_embedded.json'
    collection_name = "TestBVBb"
    persist_dir = "S:\SID\Analytics\Working Files\Individual\Florian\Projects\semantic_search\Database\ChromaDB"  # folder name for storing the persistent DB

    logger.info("Starting script...")

    # 1) Load your DataFrame
    df = prepare_dataframe(input_path, output_csv_path=None)  # No output path = no CSV is going to be saved

    # 2) Populate the persistent Chroma DB with the 1536D embeddings
    create_or_update_chroma(df, collection_name, persist_path=persist_dir)

    # 3) Query the database with some text
    query_text = "Breeding of Horses"
    # Define a similarity threshold (adjust as needed)
    similarity_threshold = 1.3

    # Query the database with the similarity threshold instead of top_n
    results_df = query_chroma(query_text, collection_name, similarity_threshold=similarity_threshold,
                              persist_path=persist_dir)

    # Logger statement for the number of retrieved entries
    retrieved_count = len(results_df)
    logger.info("Query returned %d entries with similarity threshold ≤ %.2f.", retrieved_count, similarity_threshold)

    # Display the first 10 results if available
    if not results_df.empty:
        logger.info("Head of the results:\n%s", results_df.head(10).to_string())
    else:
        logger.info("No results found matching the similarity threshold.")

    # Possibly save the results
    results_df.to_pickle("query_results.pkl")

    print("Preview of the resulting DataFrame:")
    print(results_df.head())

    results_df.sort_values(by="distance", ascending=True, inplace=False)
    print(f'top 5 results {results_df.sort_values(by="distance", ascending=True, inplace=False)[["distance", "document"]].head()} top bottom five {results_df.sort_values(by="distance", ascending=True, inplace=False)[["distance", "document"]].tail()}')


    logger.info("Done.")
