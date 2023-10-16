import os
from dotenv import load_dotenv
import pandas as pd
from haystack.nodes import EmbeddingRetriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers

prepared_dir = '../data/prepared'
prepared_recipe_file_name = 'recipes_prepared.csv'

def load_prepared_data(start_record,end_record):
    """
    Load data from file preprocessed initially
    Args:
        start_record: start records to load in case large files
        end_record: last record to load
    Returns:
        Dataframe: Panda Dataframe with columns question, answer, etc..        
    """
    prepared_recipe_file = prepared_dir+'/'+ prepared_recipe_file_name
    print('prepared file: ',prepared_recipe_file)
    print('loading data...')
    return pd.read_csv(prepared_recipe_file,index_col=False)[start_record:end_record]


def initialize_document_store():
    """
    Initialize a In Memory document store and retriever.

    Args:
        documents (List[Document]): List of documents to be stored in the document store.

    Returns:
        document_store (InMemoryDocumentStore): In Memory document store.
        retriever (EmbeddingRetriever): Embedding retriever.
    """
    
    # Initialize document store
    document_store = InMemoryDocumentStore()

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        use_gpu=False,
        scale_score=False,)
    

    return document_store, retriever 

def add_embedding_to_data(df, retriever):
    """
    Add embedded data to dataframe

    Returns:
        Dataframe: Panda Dataframe with columns embedding 
        and column question renamed as context. 
    """
    questions = list(df["question"].values)
    df["embedding"] = retriever.embed_queries(queries=questions).tolist()
    df = df.rename(columns={"question": "content"})
    return df

def load_document_store(document_store,df):
    """Load data to document store"""
    docs_to_index = df.to_dict(orient="records")
    document_store.delete_documents()
    document_store.write_documents(docs_to_index)
    print('document store loaded')
    return document_store

if __name__=="__main__":

    load_dotenv("../.env")
    openai_key = os.getenv("OPENAI_API_KEY")

    query = input("Ask a question: ")

    
    document_store, retriever = initialize_document_store()

    raw_df = load_prepared_data(0,100)
    df = add_embedding_to_data(raw_df,retriever)

    document_store = load_document_store(document_store, df)

    pipe = FAQPipeline(retriever=retriever)

    # Run any question and change top_k to see more or less answers
    result = pipe.run(query=query, params={"Retriever": {"top_k": 1}})
    ##TODO add prompt and llm here
    
    #print(result['answers'][0].answer)
    print(result)