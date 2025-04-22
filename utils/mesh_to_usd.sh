#!/bin/bash

set -e

# Define variables
INPUT="../../objects/object_id/one.obj"   # Path to the input mesh file
OUTPUT="../../objects/object_id/one.usd"  # Path to save the converted USD file
MASS="5.0"                      # Mass of the object in kg
COLLISION="convexHull"          # Collision approximation method
HEADLESS="--headless"           # Run without GUI (leave empty to enable GUI)

# Run the mesh conversion script using IsaacLab
../../../external_tools/IsaacLab/isaaclab.sh -p scripts/tools/convert_mesh.py \
  --collision-approximation "$COLLISION" \
  --mass "$MASS" \
  $HEADLESS \
  "$INPUT" \
  "$OUTPUT"


echo "Mesh successfully converted to USD: $OUTPUT"
