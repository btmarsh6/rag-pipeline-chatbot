# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None


# +
import os
import pandas as pd

# +
extract_dir = './src/data/raw'
prepared_dir = './src/data/prepared'
raw_recipe_file_name = 'RAW_recipes.csv'
prepared_recipe_file_name = 'recipes_prepared.csv'

# +
def prep_question(row):
    name = row['name']
    description = row['description']
    ingredients = row['ingredients']
    recipe =  f"name: {name}\ndescription: {description}\ningredients: {ingredients}"
    return recipe

# +
def prep_answer(row):    
    description = row['description']    
    recipe =  f"recommendation: {description}"
    return recipe

# +
# prepare df with extra columns
def prepare_df():
    """Prepare df with add columns for question/answer""" 
    raw_recipe_file = os.path.join(extract_dir, raw_recipe_file_name) 
    print(raw_recipe_file)      
    if not os.path.isfile(raw_recipe_file):        
       print(f"File not found: {raw_recipe_file}")
       return df.empty
    raw_df = pd.read_csv(raw_recipe_file)
    raw_df['question'] = raw_df.apply(prep_question, axis=1)
    raw_df['answer'] = raw_df.apply(prep_answer, axis=1)
    return raw_df

# +
# write a function that saves a dataframe to duckdb
def save_prepared(df):
    """Save prepared file to folder"""
    #TODO: use DuckDB as storage instead 
    if not os.path.exists(prepared_dir):
        os.makedirs(prepared_dir)
        
    preprocessed_file = f'{prepared_dir}/{prepared_recipe_file_name}'
    print('Preprocessed file: {preprocessed_file}')   
    df.to_csv(preprocessed_file,index= False)

# +
if __name__ == "__main__":

    df = prepare_df()
    if not df.empty:       
       save_prepared(df)
       print('Saved to preprocessed folder') 
    else:
       print('File is empty!') 
