import os
from haystack import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, PromptNode, PromptTemplate, AnswerParser
from dotenv import load_dotenv
import chainlit as cl


def initialize_rag_pipeline(index_path,config_path,openai_key):
    
    """
    Initialize a pipeline for RAG-based chatbot.
    Args:
        retriever (EmbeddingRetriever): Embedding retriever.
        openai_key (str): API key for OpenAI.
    Returns:
        query_pipeline (Pipeline): Pipeline for RAG-based question answering.
    """
    # load faiss index
    try:
        document_store = FAISSDocumentStore.load(index_path=index_path, config_path=config_path)
    except:
        print("Error loading fais index")
        return None

    prompt_template = PromptTemplate(prompt=""""Generate the recipe Steps by the Ingredients and follow the similar order as provided in the Examples\n
                                                Ingredients: {query}\n
                                                Examples: {join(documents)}
                                                Steps:
                                            """,
                                            output_parser=AnswerParser())
    
    prompt_node = PromptNode(model_name_or_path="gpt-4",
                             api_key=openai_key,
                             default_prompt_template=prompt_template )
    
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2")

    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])
    return query_pipeline
 
  


# load env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

#paths to index and config
index_path = os.path.join(os.getcwd(),"qa_index.faiss")
config_path = os.path.join(os.getcwd(), "qa_config.json")

query_pipeline = initialize_rag_pipeline(index_path,config_path,openai_api_key)

@cl.on_message
async def main(message: str):
    # Use pipeline to get a response
    output = query_pipeline.run(query=message)

    # Create Chainlit message with response
    response = output['answers'][0].answer
    msg = cl.Message(content=response)

    # Send message to the user
    await msg.send()