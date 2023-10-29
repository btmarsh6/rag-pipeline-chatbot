from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, AnswerParser, PromptNode, PromptTemplate
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

    if "question" not in df or  "answer" not in df:
            raise ValueError("The CSV must contain two columns named 'question' and 'answer'")    

    df = df.rename(columns={"answer": "content"})
    
    docs_dicts = df.to_dict(orient="records")

    docs = []
    for dictionary in docs_dicts:            
        docs.append(Document.from_dict(dictionary))

    return docs


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
    document_store = FAISSDocumentStore(faiss_index_factory_str='Flat', return_embedding=True)

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


def initialize_rag_pipeline(retriever, openai_key):
    """
    Initialize a pipeline for RAG-based chatbot.
    Args:
        retriever (DensePassageRetriever): Dense passage retriever.
        openai_key (str): API key for OpenAI.
    Returns:
        query_pipeline (Pipeline): Pipeline for RAG-based question answering.
    """
    #prompt_template = PromptTemplate(prompt=""""Answer the following query based on the provided context. If the context does
    #                                            not include an answer, reply with 'The data does not contain information related to the question'.\n
    #                                            Query: {query}\n
    #                                            Documents: {join(documents)}
    #                                            Answer: 
    #                                        """,
    #                                        output_parser=AnswerParser())

    prompt_template = PromptTemplate(prompt=""""Generate the recipe Steps by the Ingredients and follow the similar order as provided in the Examples\n
                                                Ingredients: {query}\n
                                                Examples: {join(documents)}
                                                Steps:
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
    openai_key = os.environ['OPENAI_API_KEY']    

    # Initialize documents
    #documents = initialize_documents('data/recipes_prepared_100.csv')

    # Initialize document store and retriever
    #document_store, retriever = initialize_faiss_document_store(documents=documents)

    # Initialize pipeline
    #query_pipeline = initialize_rag_pipeline(retriever=retriever, openai_key=openai_key)



# generator = RAGenerator(
#     model_name_or_path="facebook/rag-token-nq",
#     use_gpu=True,
#     top_k=1,
#     max_length=200,
#     min_length=2,
#     embed_title=True,
#     num_beams=2,
# )
