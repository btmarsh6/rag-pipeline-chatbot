import pandas as pd
import kaggle
from zipfile import ZipFile
from ast import literal_eval

# Download data
print('Downloading data from Kaggle...')
kaggle.api.authenticate()
kaggle.api.dataset_download_file(dataset='shuyangli94/food-com-recipes-and-user-interactions/',
                                 file_name='RAW_recipes.csv',
                                 path='../../data/')

# Unzip data file
with ZipFile('../../data/RAW_recipes.csv.zip', 'r') as zip:
    zip.extractall('../../data/')

# Import data to pandas
print('Processing data...')
raw_df = pd.read_csv('../../data/RAW_recipes.csv',
                     converters={'tags': literal_eval,
                                 'nutrition': literal_eval,
                                 'steps': literal_eval,
                                 'ingredients': literal_eval})


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
    minutes = row['minutes']
    tags = ', '.join(row['tags'])
    n_ingredients = row['n_ingredients']
    ingredients = ', '.join(row['ingredients'])
    steps = numbered_steps(row['steps'])    
    nutrition_info = nutrition_labels(row['nutrition'])
    
    # Combine fields into full recipe    
    full_recipe = f"Name: {name}\nCook Time: {minutes} minutes\nTags: {tags}\nNumber of ingredients: {n_ingredients}\nIngredients List: {ingredients}\nSteps:\n{steps}\nNutrition: {nutrition_info}"
    return full_recipe


# Create single full_recipe doc for each row.
raw_df['full_recipe'] = raw_df.apply(create_doc, axis=1)

# Create CSV of full recipe documents
print('Saving prepared documents...')
recipe_df = pd.DataFrame(raw_df['full_recipe'])
recipe_df.to_csv('../../data/recipe_docs.csv', index=False)
