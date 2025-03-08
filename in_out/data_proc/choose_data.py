import os
import shutil
import data.ShapeNetCore as ShapeNetCore # Add this project to PATH
import numpy as np
from tqdm import tqdm
from glob import glob
import json
"""Script to choose the data from ShapeNetCore and move them to ShapeNetCoreURDF"""
"""Modified from TouchSDF, Bingyao, 3/6/2025"""

if __name__=='__main__':
    # Set directories
    data_dir = os.path.dirname(ShapeNetCore.__file__)

    output_dir = os.path.join(os.path.dirname(ShapeNetCore.__file__), '..', 'ShapeNetCore')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Load the categories
    with open('datasets/categories_root.json', 'r') as file:
        categories_dict = json.load(file)

    # Select up to 200 folders inside each category from ShapeNetCore
    for category in tqdm(categories_dict.keys()):
        # Get the list of all the models, excluding hidden files
        models = [os.path.basename(a) for a in glob(os.path.join(data_dir, categories_dict[category], '*')) if not os.path.basename(a).startswith('.')]
        print(f'Category {category}: {len(models)} models')

        # Limit the number of models to 200 if more exist
        num_models = min(len(models), 200)
        if len(models) > num_models:
            models = np.random.choice(models, num_models, replace=False)

        # Process each model
        for model in models:
            model_path = os.path.join(data_dir, categories_dict[category], model, 'models')
            if os.path.exists(model_path) and 'model_normalized.obj' in os.listdir(model_path):
                dest_path = os.path.join(output_dir, categories_dict[category], model)
                shutil.copytree(os.path.join(data_dir, categories_dict[category], model), dest_path)