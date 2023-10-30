# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None

# -
import os
import pandas as pd
import numpy as np
import duckdb
import kaggle
from ast import literal_eval

# +
dataset = 'shuyangli94/food-com-recipes-and-user-interactions'
extract_dir = './src/data/raw'
prepared_dir = './src/data/prepared'
raw_recipe_file_name = 'RAW_recipes.csv'
raw_interactions_file_name = 'RAW_interactions.csv'
prepared_recipe_file_name = 'recipes_prepared.csv'
raw_recipe_file = os.path.join(extract_dir, raw_recipe_file_name) 
raw_interactions_file = os.path.join(extract_dir,raw_interactions_file_name)


# +
def download_data_from_kaggle(raw_recipe_file,raw_interactions_file):
    """Download data from kaggle"""     
    print(raw_recipe_file)      
    print(raw_interactions_file) 
    if not os.path.isfile(raw_recipe_file) and not os.path.isfile(raw_recipe_file):
        try:       
           print("Dataset not found, using kaggle-api tool for download")
           # Download the data
           kaggle.api.authenticate()
           kaggle.api.dataset_download_files(dataset, path=extract_dir, unzip=True)
        except Exception as e:
           print("Not able to download from kaggle, err: ", e)
           return False
    return True  

# +
def merge_recipe_and_interactions(raw_recipe_file,raw_interactions_file):   
    recipe_df = pd.read_csv(raw_recipe_file,
                        converters={'tags': literal_eval,
                                    'nutrition': literal_eval,
                                    'steps': literal_eval,
                                    'ingredients': literal_eval})

    interactions_df = pd.read_csv(raw_interactions_file)

    # Calculate average rating for each recipe. Add this to recipe_df
    avg_rating = interactions_df.groupby(by='recipe_id', as_index=False)['rating'].agg(np.mean)
    recipe_df = pd.merge(recipe_df, avg_rating, how='left', left_on='id', right_on="recipe_id")
    
    return recipe_df

# +
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

# +
def numbered_steps(steps):
    """ (list) -> str
    Takes list of steps and converts to a string of numbered steps. 
    """
    result = ''
    newline = '\\n'
    for i, step in enumerate(steps):
        result += f"{i+1} {step} {newline} "
    return result

# +
def to_tags(row):
    return ', '.join(row['tags'])

# +
def to_ingredients(row):
    return ', '.join(row['ingredients'])

# +
def to_rating(row):
    return round(row['rating'], 1)

# +
def to_steps(row):
    return numbered_steps(row['steps'])

# +
def to_nutrition(row):
    return nutrition_labels(row['nutrition'])

# +
def create_full_recipe(row):
    """
    Collapses each row into a single document for the recipe,
    adding labels to column values.
    """
    # Extract and format relevant data from each field
    name = row['name']
    rating = row['rating']
    minutes = row['minutes']
    tags = row['tags']
    description = row['description']
    n_ingredients = row['n_ingredients']
    ingredients = row['ingredients']
    steps = row['steps']    
    nutrition_info = row['nutrition']

    # Combine fields into full recipe    
    full_recipe = f"Name: {name}\n\nRating: {rating}/5\n\nCook Time: {minutes} minutes\n\nTags: {tags}\n\nDescription: {description}\n\nNumber of ingredients: {n_ingredients}\n\nIngredients List: {ingredients}\n\nSteps:\n{steps}\n\nNutrition: {nutrition_info}"
    return full_recipe

# +
def df_transform(df):
    df['rating'] = df.apply(to_rating, axis=1)
    df['tags'] = df.apply(to_tags, axis=1)    
    df['nutrition'] = df.apply(to_nutrition, axis=1)    
    df['ingredients'] = df.apply(to_ingredients, axis=1)
    df['steps'] = df.apply(to_steps, axis=1)    
    df['full_recipe'] = df.apply(create_full_recipe, axis=1)
    return df

# +
def final_rename(df):     
    df.fillna(value="", inplace=True)
    df["name"] = df["name"].apply(lambda x: x.strip())

    # use only ingredients and steps
    #df['question'] = "Ingredients:" + df['ingredients']

    # use full recipe as question
    df['question'] = df['full_recipe']
    df['answer'] = "Ingredients:"+ df['ingredients']+"\nSteps:"+ df['steps'] 
    return df

# +
# write a function that saves a dataframe to duckdb
def save_to_duckdb(df, table_name, db_path):
    """Save dataframe to duckdb"""
    conn = duckdb.connect(db_path)
    conn.register('df', df)
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    conn.close()

# +
# write a function that saves a dataframe to duckdb
def save_prepared(df,dir, file_name):
    """Save prepared file to folder"""
    #TODO: use DuckDB as storage instead 
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    preprocessed_file = f'{dir}/{file_name}'
    print(f'Preprocessed file: {preprocessed_file}')   
    df.to_csv(preprocessed_file,index= False)

# +
if __name__ == "__main__":

    print('Processing data...')
    isDownloaded = download_data_from_kaggle(raw_recipe_file,raw_interactions_file)
    if isDownloaded:       
       df = merge_recipe_and_interactions(raw_recipe_file,raw_interactions_file)
       if df.empty:
           print('DataFrame is empty!')
       else:
           df_transformed = final_rename(df_transform(df))           

           save_prepared(df_transformed,prepared_dir, prepared_recipe_file_name)
           print('Saved to preprocessed folder') 
           
    else:
       print('Issue with kaggle download') 
    




