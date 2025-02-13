import os
import json
import pandas as pd
root = r'S:\SID\Analytics\Working Files\Individual\Florian\Projects\semantic_search\Data'
source = r'db_embedded.json'

input = os.path.join(root, source)

with open(input, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert the JSON data into a DataFrame
df = pd.DataFrame(data)

print(df.head())