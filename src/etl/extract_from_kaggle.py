import os
import kaggle
import pandas as pd
import zipfile

## cd to src 


user = os.getenv('kaggle_username')
api_key = os.getenv('kaggle_key')
# print(api_key)

def download_datasets(extract_path):
    # create a raw_data dir to save raw data from kaggle
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)


    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('shuyangli94/food-com-recipes-and-user-interactions', path='./raw_data/', unzip=True) # here download zip

    zip_file = os.path.join('./raw/', "food-com-recipes-and-user-interactions.zip")
    print(zip_file)

    # Unzip the downloaded dataset
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extract the raw data successfully!")


def merge_csv(df1, df2, key, method, save_path):
    merged_df = pd.merge(df1, df2, on = key, how = method).drop_duplicates(subset=[key], keep='first')

    merged_df.to_csv(f'{save_path}', index=False)
    return merged_df 

def get_dict_for_nutrition(row):
    import ast

    # here df['nutrition] originally is str -> we have to convert it to list
    nutrition_names = ['calories', 'total_fat', 'sugar', 'sodium', 'protein', 'saturated_fat'] 
    row = ast.literal_eval(row)
    return {nutrition_names[i]: row[i] for i in range(len(nutrition_names))}

def prepare_data(path, merged_save_path):
    merged_raw_data = pd.read_csv(f'{path}/{merged_save_path}')

    # here we'll do some steps 
    # 1. convert nutrition to a dict 
    # Nutrition information (calories (#), total fat (PDV), sugar (PDV) , sodium (PDV) , protein,  saturated fat

    merged_raw_data['nutrition'] = merged_raw_data['nutrition'].apply(get_dict_for_nutrition)
    sample = merged_raw_data.head(5)
    sample.to_csv(f'{path}/sample.csv', index=False)
    merged_raw_data.to_csv(f'{path}/{merged_save_path}', index=False)
    return merged_raw_data


def save_to_DuckDB(df_path, table_name):
    import duckdb

    df = pd.read_csv(df_path)

    conn = duckdb.connect()
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")










# merge_csv(df1 = 'RAW_recipes.csv', df2 = 'RAW_interactions.csv', key = , method, save_path)
extract_path = './raw_data/'
path = './raw_data'
df1 = 'RAW_recipes.csv'
df2 = 'RAW_interactions.csv'
merged_save_path = 'merged_raw_data_to_use.csv'

pd.set_option('display.max_columns', None)
data1 = pd.read_csv(f'{path}/{df1}')
data2 = pd.read_csv(f'{path}/{df2}')

data1.rename(columns={"id": "recipe_id"}, inplace=True)
# the del col below, I specific data1 and data2 - be careful when run these lines
del data1['contributor_id'] # we dont need contributor_id and submitted_date
del data1['submitted']
del data2['user_id']
del data2['date']

print(f"We'll keep columns {data1.columns} for dataset1 - {df1}")
print(f" We'll keep {data2.columns} for dataset2 - {df2}")

print(f'{path}/{merged_save_path}')

# now we'll merge 2 csvs
merge_csv(df1 = data1, df2 = data2, key = 'recipe_id', method = 'left', save_path = f'{path}/{merged_save_path}')
# then read the merged one as our raw data file
prepare_data(path, merged_save_path)

# save in DuckDB
df_path = f'{path}/{merged_save_path}'
save_to_DuckDB(df_path, table_name= 'raw_data_for_chatbot')