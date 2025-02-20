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
    Upsert data into Chroma without duplicates:
      1) If the collection doesn't exist, create it and insert ALL rows.
      2) If it does exist, skip any IDs that are already there, and insert only new rows.
         (No duplicates, no warnings).

    DataFrame columns:
      - 'pp_id' (unique ID for each row)
      - 'embedding' (list of floats or JSON string)
      - 'sentence' (optional; will be stored as 'documents')
      - plus other columns for metadata

    Args:
        df (pd.DataFrame): Data with 'pp_id' and 'embedding' columns (and optional 'sentence').
        collection_name (str): The name of the Chroma collection.
        persist_path (str): Folder path for persistent Chroma storage.
        batch_size (int): Number of rows to insert per batch (default=100).
    """
    logger.info("Initializing persistent Chroma client via `PersistentClient`.")
    chroma_client = chromadb.PersistentClient(path=persist_path)

    # 1) Check if collection exists
    existing_collections = chroma_client.list_collections()
    collection_exists = (collection_name in existing_collections)

    # 2) Create or get collection
    if not collection_exists:
        logger.info(f"Collection '{collection_name}' does NOT exist. Creating new collection.")
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "hnsw:search_ef": 100}
        )
    else:
        logger.info(f"Collection '{collection_name}' already exists. Retrieving it.")
        collection = chroma_client.get_collection(name=collection_name)

    # 3) Ensure 'embedding' is a list of floats
    def ensure_list(val):
        if isinstance(val, str):
            return json.loads(val)  # e.g. "[0.123, 0.456]" -> [0.123, 0.456]
        return val

    df["embedding"] = df["embedding"].apply(ensure_list)

    # 4) Prepare all IDs, embeddings, etc.
    all_ids = df["pp_id"].astype(str).tolist()
    all_embeddings = df["embedding"].tolist()

    # If there's a 'sentence' column, use it for documents; otherwise use empty strings
    documents = (
        df["sentence"].fillna("").astype(str).tolist()
        if "sentence" in df.columns else [""] * len(df)
    )

    # Exclude 'pp_id' and 'embedding' from metadata
    exclude_cols = {"pp_id", "embedding"}
    meta_columns = [col for col in df.columns if col not in exclude_cols]

    metadatas = []
    for _, row in df.iterrows():
        meta = {}
        for col in meta_columns:
            val = row[col]
            # If it's a list, store it as JSON
            if isinstance(val, list):
                val = json.dumps(val)
            meta[col] = val
        metadatas.append(meta)

    # 5) If collection already existed, figure out which IDs are new
    if collection_exists:
        logger.info(f"Checking which of the {len(all_ids)} IDs already exist in the collection.")
        # Query the DB for these IDs (include=[] means we don't fetch documents, embeddings, etc.)
        existing_data = collection.get(ids=all_ids, include=[])
        found_ids = set(existing_data["ids"]) if existing_data["ids"] else set()

        # Filter out the rows whose IDs are already in the collection
        new_ids = []
        new_embeddings = []
        new_docs = []
        new_metas = []
        for i, pid in enumerate(all_ids):
            if pid not in found_ids:
                new_ids.append(pid)
                new_embeddings.append(all_embeddings[i])
                new_docs.append(documents[i])
                new_metas.append(metadatas[i])

        # If no new rows, we're done
        if not new_ids:
            logger.info("All IDs already exist in the collection. No new data to insert.")
            return

        logger.info(f"{len(found_ids)} IDs already existed, {len(new_ids)} are new. Inserting only the new ones.")
        # Insert new items in batches
        _insert_in_batches(
            collection,
            new_ids,
            new_embeddings,
            new_docs,
            new_metas,
            batch_size,
            collection_name
        )

    else:
        # Collection is newly created, so everything in df is new
        logger.info(f"Collection is new. Inserting all {len(df)} rows.")
        _insert_in_batches(
            collection,
            all_ids,
            all_embeddings,
            documents,
            metadatas,
            batch_size,
            collection_name
        )

    logger.info(f"Upsert complete. Collection '{collection_name}' updated with no duplicates.")


def _insert_in_batches(collection, ids, embeddings, documents, metadatas, batch_size, collection_name):
    """
    Helper to insert data in batches into a Chroma collection.
    """
    total = len(ids)
    for start_idx in range(0, total, batch_size):
        end_idx = start_idx + batch_size
        sub_ids = ids[start_idx:end_idx]
        sub_emb = embeddings[start_idx:end_idx]
        sub_docs = documents[start_idx:end_idx]
        sub_metas = metadatas[start_idx:end_idx]

        collection.add(
            ids=sub_ids,
            embeddings=sub_emb,
            documents=sub_docs,
            metadatas=sub_metas
        )
        logging.info(f"Inserted rows {start_idx} to {end_idx - 1} into '{collection_name}'.")



# Function for querying the ChromaDB and returning the results
# extended with optional metadata filter (where clause)
def query_chroma(
    query_text: str,
    collection_name: str,
    similarity_threshold: float = 0.54,
    initial_top_n: int = 999999999,         # probably just setting this to a high number is fine as long as the dataset is not too big
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
