import os
import json
root = r'S:\SID\Analytics\Working Files\Individual\Florian\Projects\semantic_search\Data'
source = r'db_embedded.json'

input = os.path.join(root, source)

with open(input, "r", encoding="utf-8") as f:
    data = json.load(f)

