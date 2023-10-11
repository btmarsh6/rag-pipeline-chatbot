import pandas as pd
import numpy as np
import kaggle
from ast import literal_eval

# Download data
print('Downloading data from Kaggle...')
kaggle.api.authenticate()
kaggle.api.dataset_download_files(dataset='shuyangli94/food-com-recipes-and-user-interactions/',
                                  path='../../data/',
                                  unzip=True)

# Import data to pandas
print('Processing data...')
recipe_df = pd.read_csv('../../data/RAW_recipes.csv',
                        converters={'tags': literal_eval,
                                    'nutrition': literal_eval,
                                    'steps': literal_eval,
                                    'ingredients': literal_eval})

interactions_df = pd.read_csv('../../data/RAW_interactions.csv')

# Calculate average rating for each recipe. Add this to recipe_df
avg_rating = interactions_df.groupby(by='recipe_id', as_index=False)['rating'].agg(np.mean)
recipe_df = pd.merge(recipe_df, avg_rating, how='left', left_on='id', right_on="recipe_id")


def nutrition_labels(nutrition):
    """ (list) -> str
    Takes list of nutrition information from dataframe and adds labels.
    """
    nutrition_labels = [' calories', '% total fat', '% sugar',
                        '% sodium', '% protein', '% saturated fat',
                        '% total carbohydrates']
    result = ''
    for i in range(6):
        labelled = f'{nutrition[i]}{nutrition_labels[i]}, '
        result += labelled
    return result


def numbered_steps(steps):
    """ (list) -> str
    Takes list of steps and converts to a string of numbered steps. 
    """
    result = ''
    for i, step in enumerate(steps):
        result += f'{i+1} {step}\n'
    return result


def create_doc(row):
    """
    Collapses each row into a single document for the recipe,
    adding labels to column values.
    """
    # Extract and format relevant data from each field
    name = row['name']
    rating = round(row['rating'], 1)
    minutes = row['minutes']
    tags = ', '.join(row['tags'])
    description = row['description']
    n_ingredients = row['n_ingredients']
    ingredients = ', '.join(row['ingredients'])
    steps = numbered_steps(row['steps'])    
    nutrition_info = nutrition_labels(row['nutrition'])

    # Combine fields into full recipe    
    full_recipe = f"Name: {name}\n\nRating: {rating}/5\n\nCook Time: {minutes} minutes\n\nTags: {tags}\n\nDescription: {description}\n\nNumber of ingredients: {n_ingredients}\n\nIngredients List: {ingredients}\n\nSteps:\n{steps}\n\nNutrition: {nutrition_info}"
    return full_recipe


# Create single full_recipe doc for each row.
recipe_df['full_recipe'] = recipe_df.apply(create_doc, axis=1)

# Create CSV of full recipe documents
print('Saving prepared documents...')
docs_df = pd.DataFrame(recipe_df[['name', 'full_recipe']])
docs_df.to_csv('../../data/recipe_docs.csv', index=False)
