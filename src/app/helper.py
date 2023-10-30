from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, AnswerParser
from haystack.nodes import PromptNode, PromptTemplate
from haystack import Pipeline
import os
import pandas as pd


def initialize_documents(file_path):
    """
    Casts recipes from prepared recipe_docs.csv file into document structure for Haystack.

    Args:
        file_path (str): location of recipe_docs.csv file
    Returns:
        documents ()
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Small sample for testing purposes.
    sample_df = df.sample(n=1000)

    # Cast data into Haystack Document objects
    titles = list(sample_df['name'].values)
    texts = list(sample_df['full_recipe'].values)
    documents = []
    for title, text in zip(titles, texts):
        documents.append(Document(content=text, meta={'name': title or ''}))
    return documents


def initialize_faiss_document_store(documents):
    """
    Initialize FAISS document store and retriever.
    Args:
        documents (list): List of documents to be stored in document store.
    Returns:
        document_store (FAISSDocumentStore): FAISS document store.
        retriever (DensePassageRetriever): Dense passage retriever
    """
    # Initialize DocumentStore
    document_store = FAISSDocumentStore(faiss_index_factory_str='Flat',
                                        return_embedding=True,
                                        similarity='dot_product')

    # Initialize Retriever
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
        embed_title=True
        )

    # Delete existing documents in document store
    document_store.delete_documents()
    document_store.write_documents(documents)

    # Add documents embeddings to index
    document_store.update_embeddings(retriever=retriever)

    return document_store, retriever


def preloaded_faiss(index_path, config_path):
    """
    Initialize FAISS document store and retriever using pre-calculated index.
    """

    # Initialize DocumentStore
    try:
        document_store = FAISSDocumentStore.load(index_path=index_path,
                                                 config_path=config_path)
    except:
        print("Error initializing document store.")
        return None

    # Initialize Retriever
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False,
        embed_title=True
        )

    return document_store, retriever


def initialize_rag_pipeline(retriever, openai_key):
    """
    Initialize a pipeline for RAG-based chatbot.
    Args:
        retriever (DensePassageRetriever): Dense passage retriever.
        openai_key (str): API key for OpenAI.
    Returns:
        query_pipeline (Pipeline): Pipeline for RAG-based question answering.
    """
    prompt_template = PromptTemplate(prompt=""""Offer the user the recipe that best matches their query.
                                     If they ask for a different option, provide them the next best match.
                                     Related text: {join(documents)} \n\n Question: {query} \n\n Answer:
                                            """,
                                            output_parser=AnswerParser())
    prompt_node = PromptNode(model_name_or_path="gpt-3.5-turbo",
                             api_key=openai_key,
                             default_prompt_template=prompt_template,
                             max_length=500,
                             model_kwargs={"stream": True})

    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

    return query_pipeline


if __name__ == "__main__":
    # Load environment variables (if any)
    openai_key = os.environ['OPENAI_HACKTOBERFEST_KEY']

    # Initialize documents
    # documents = initialize_documents('../../data/recipe_docs.csv')

    # Initialize document store and retriever
    # document_store, retriever = initialize_faiss_document_store(documents=documents)
    document_store, retriever = preloaded_faiss(index_path=index_path, config_path=)

    # Initialize pipeline
    query_pipeline = initialize_rag_pipeline(retriever=retriever, openai_key=openai_key)

