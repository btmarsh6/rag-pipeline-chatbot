import pandas as pd
import ast 
import duckdb
from pprint import pprint

path = './raw_data'
table_name= 'raw_data_for_chatbot'

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
data = pd.read_csv(f'{path}/merged_raw_data_to_use.csv') # [51.5, 0.0, 13.0, 0.0, 2.0, 0.0, 4.0]
# print(data.head(5))

df = data

conn = duckdb.connect()
conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")

result = conn.execute(f"SELECT * FROM df LIMIT 1")
for row in result.fetchall():
    pprint(row)
