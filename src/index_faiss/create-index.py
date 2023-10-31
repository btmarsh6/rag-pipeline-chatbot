# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None

# -

import os
import pandas as pd
from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import  EmbeddingRetriever


prepared_data_dir = 'src/data/prepared/'
prepared_recipe_file_name = 'recipes_prepared.csv'

def delete_file(file_path):
    """Delete file in directory"""
    try:
        os.remove(file_path)
    except:
        pass


def clean_files(document_store_dir):
    """ Clean all files to recreate index again
        Returns:
        paths: faiss_documents_store.db, qa_config.json, qa_index.faiss
         """
    try:
        # Clean files related to FAISS, config and index config
        index_path = os.path.join(document_store_dir, "qa_index.faiss")
        config_path = os.path.join(document_store_dir, "qa_config.json")
        print('index_path',index_path)
        db_base = os.path.join(document_store_dir, "faiss_document_store.db")
  
        delete_file(index_path)
        delete_file(config_path)
        delete_file(db_base)

        return index_path,config_path
    except:
        pass


def load_data(file_path,start_record,end_record):
    """
    Load recipes from prepared recipe_docs.csv file into document structure for Haystack.

    Args:
        file_path (str): location of recipe_docs.csv file
    Returns:
        documents ()
    """
    # Load data
    df = pd.read_csv(file_path,index_col=False)[start_record:end_record]  

    if "question" not in df or  "answer" not in df:
            raise ValueError("The data must contain two columns named 'question' and 'answer'")    

    df = df.rename(columns={"answer": "content"})
    
    docs_dicts = df.to_dict(orient="records")

    docs = []
    for dictionary in docs_dicts:            
        docs.append(Document.from_dict(dictionary))

    return docs


def initialize_document_store():
    """
    Initialize a FAISS document store and retriever.

    Args:
        documents (List[Document]): List of documents to be stored in the document store.

    Returns:
        document_store (FAISSDocumentStore): FAISS document store.
        retriever (EmbeddingRetriever): Embedding retriever.
    """
    
    # Initialize DocumentStore
    document_store = FAISSDocumentStore(faiss_index_factory_str='Flat',embedding_dim=384, 
                                        return_embedding=True)

    # Initialize Retriever
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    

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
    
    document_store.delete_documents()
    document_store.write_documents(df)
    print('document store loaded')
    return document_store

def create_index(data_dir, document_store_dir):
    """ Create FAISS index and embedding and store index in directory
    """    
   
    try:
       # clean files
       index_path,config_path = clean_files(document_store_dir)    

       # initialize index
       document_store, retriever = initialize_document_store()

       # load data
       data_file = data_dir + prepared_recipe_file_name
       df = load_data(data_file,0,100) 
       print("data size", len(df) )  

       # load document store
       document_store = load_document_store(document_store, df)
       

       # add embeddings
       document_store.update_embeddings(retriever)

       # save index
       document_store.save(index_path=index_path, config_path=config_path)
    except Exception as e:
       print("error",e)

if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(),prepared_data_dir)
    document_store_dir = os.getcwd()
    print('data_dir',data_dir)

    create_index(data_dir, document_store_dir)

    print('index create completed')