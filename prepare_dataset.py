import json
import pandas as pd
import numpy as np
import umap
import os

def prepare_dataframe(
    input_json_path: str,
    output_csv_path: str = None
) -> pd.DataFrame:
    """
    1) Load JSON data into a Pandas DataFrame.
    2) Rename 'embedding' -> 'embedding_long'.
    3) Reduce embedding dimensions to 25 via UMAP, save as 'embedding_short'.
    4) Generate ID: 'steam_1', 'steam_2', ...
    5) (Optional) Write the resulting DataFrame to a CSV file.
    """

    # 1) Load JSON data
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # data is expected to be a list of dicts

    df = pd.DataFrame(data)

    # 2) Rename column 'embedding' -> 'embedding_long'
    df.rename(columns={"embedding": "embedding_long"}, inplace=True)

    # 3) Use UMAP to reduce embedding dimensions to 25
    #    - First convert 'embedding_long' to a matrix of shape [num_rows, original_dim]
    embeddings_matrix = np.vstack(df["embedding_long"].values)
    reducer = umap.UMAP(n_components=25, random_state=42)
    embedding_25d = reducer.fit_transform(embeddings_matrix)

    # embedding_25d is now a NumPy array of shape [num_rows, 25].
    # Convert each row back to a Python list so it can be stored easily in the DataFrame
    df["embedding_short"] = embedding_25d.tolist()

    # 4) Generate an incremental ID: steam_1, steam_2, ...
    df["id"] = [f"steam_{i+1}" for i in range(len(df))]

    # Optionally write to CSV
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)

    return df

if __name__ == "__main__":
    # Example usage:
    input_path = r"S:\SID\Analytics\Working Files\Individual\Florian\Projects\semantic_search\Data\db_embedded.json"
    output_path = r"S:\SID\Analytics\Working Files\Individual\Florian\Projects\semantic_search\Data\db_embedded_prepared.csv"

    df_prepared = prepare_dataframe(input_path, output_csv_path=output_path)

    print("Preview of the resulting DataFrame:")
    print(df_prepared.head())
    print("\nNumber of rows:", len(df_prepared))
