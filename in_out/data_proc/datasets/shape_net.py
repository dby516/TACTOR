"""
ShapeNet.
"""
import json

def extract_categories(nodes):
    """Recursively extract category id and primary label from taxonomy nodes."""
    mapping = {}
    for node in nodes:
        # Get the category id from the "metadata" field
        cat_id = node["metadata"]["name"]
        # Use the first label as the primary category label
        primary_label = node["metadata"]["label"].split(",")[0].strip()
        mapping[cat_id] = primary_label
        
        # Recursively extract from children nodes if present
        # if "children" in node:
        #     mapping.update(extract_categories(node["children"]))
    return mapping

# Load the taxonomy JSON file
with open('shapenetcore.taxonomy.json', 'r') as file:
    taxonomy_data = json.load(file)

# Build the dictionary
synth_id_to_category_dict = extract_categories(taxonomy_data)

# Write the dictionary to a JSON file
with open('categories_root.json', 'w') as outfile:
    json.dump(synth_id_to_category_dict, outfile, indent=4)
