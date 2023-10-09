# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None


# +
import os
import pandas as pd
import duckdb
import kaggle

# +
dataset = 'shuyangli94/food-com-recipes-and-user-interactions'
extract_dir = './src/data/raw'
raw_recipe_file_name = 'RAW_recipes.csv'

# +
def download_data_from_kaggle():
    """Download data from kaggle""" 
    raw_recipe_file = os.path.join(extract_dir, raw_recipe_file_name) 
    print(raw_recipe_file)      
    if not os.path.isfile(raw_recipe_file):        
       print("Dataset not found, using kaggle-api tool for download")
       # Download the data
       kaggle.api.authenticate()
       kaggle.api.dataset_download_files(dataset, path=extract_dir, unzip=True)
    return pd.read_csv(raw_recipe_file)

# +
# write a function that saves a dataframe to duckdb
def save_to_duckdb(df, table_name, db_path):
    """Save dataframe to duckdb"""
    conn = duckdb.connect(db_path)
    conn.register('df', df)
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    conn.close()

# +
if __name__ == "__main__":

    df = download_data_from_kaggle()
    if not df.empty:       
       # Save to duckdb
       db_path = "data.duckdb"
       table_name = "RAW_recipes"
       save_to_duckdb(df, table_name, db_path)
       print('Saved RAW_recipes to duckdb.') 
    else:
       print('DataFrame is empty!') 
