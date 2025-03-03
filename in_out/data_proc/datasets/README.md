## ShapeNetCore Dataset

The ShapeNetCore dataset is organized into a well-defined folder structure. Here’s an outline of the key directories and files:

```
data/ShapeNetCore/
    02691156/ 
        instance_folder_0/ 
            images/
                texture0.jpg, texture1.jpg, texture2.jpg
            models/
                model_normalized.json
                model_normalized.mtl
                model_normalized.obj
        instance_folder_1/...
        instance_folder_2/...
```

### Explanation

- **02691156/**
  An example category directory (e.g., for "airplane"). This directory contains several subdirectories, each representing a distinct model instance.

- **instance_folder/**
  A folder representing a specific model instance within the category. The folder name is a unique identifier for that instance.

- **images/**
  Contains texture images (e.g., `texture0.jpg`, `texture1.jpg`, etc.) used for rendering the 3D model.

- **models/**
  Contains the 3D model files:

  - model_normalized.json:

    A JSON file that provides metadata about the 3D model. An example content is:

    ```json
    {
      "max": [109.808, 58.4216, -87.6625],
      "centroid": [84.48248071331106, 50.21447665635025, -111.35764793156288],
      "id": "1a23fdbb1b6d4c53902c0a1a69e25bd9",
      "numVertices": 12449,
      "min": [51.4907, 0.0, -150.198]
    }
    ```

    This file includes:

    - **max:** The maximum coordinate values along the x, y, and z axes.
    - **min:** The minimum coordinate values.
    - **centroid:** The center of the model.
    - **id:** A unique identifier for the model.
    - **numVertices:** The number of vertices in the 3D mesh.

  - **model_normalized.mtl & model_normalized.obj:**
    These files represent the material definitions and the 3D geometry, respectively, in standardized formats for 3D models.



## ShapeNetCore Taxonomy JSON 

This JSON file represents a hierarchical taxonomy of object categories. Each element in the top-level array corresponds to a category (or node) in the taxonomy tree. The file is structured to include metadata, descriptive attributes, and any sub-categories (children) for each node.

```json
// Example
[{
    "metadata": {
        "numInstances": 4045,
        "name": "02691156",
        "numChildren": 11,
        "label": "airplane,aeroplane,plane"
    },
    "li_attr": {
        "id": "02691156",
        "title": "an aircraft that has a fixed wing and is powered by propellers or jets; 'the flight was delayed due to trouble with the airplane'  \n"
    },
    "children": [{...}, {...}, ...],
    "text": "airplane,aeroplane,plane(11,4045)"
},
// Additional categories
]

```

### Structure Overview

Each node in the JSON array typically contains the following fields:

- **metadata**:
  Contains key statistical and identification information for the category.
  - **numInstances**: The total number of instances for this category.
  - **name**: A unique identifier for the category.
  - **numChildren**: The number of sub-categories (children) under this category.
  - **label**: A comma-separated string of labels describing the category.
- **li_attr**:
  Provides additional attributes used for presentation or further details.
  - **id**: An identifier matching the category name.
  - **title**: A descriptive text that gives more context about the category.
- **children** (optional):
  A list of child nodes, each following the same structure as the parent node. These represent sub-categories within the taxonomy.
- **text**:
  A brief summary string that combines the label and additional information such as the number of children and instances. For example, `"airplane,aeroplane,plane(11,4045)"` indicates 11 sub-categories and 4045 instances for the airplane category.
