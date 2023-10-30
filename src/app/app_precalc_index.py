import chainlit as cl
from helper import preloaded_faiss, initialize_rag_pipeline
import os


# Load environment variables
openai_key = os.environ['OPENAI_HACKTOBERFEST_KEY']
index_path = 'rag_faiss_index.faiss'
config_path = 'rag_faiss_index.json'

# # Initialize documents
# filepath = '../../data/recipe_docs.csv'
# documents = initialize_documents(filepath)

# Initialize document store and retriever
document_store, retriever = preloaded_faiss(index_path=index_path,
                                            config_path=config_path)

# Initialize pipeline
query_pipeline = initialize_rag_pipeline(retriever=retriever,
                                         openai_key=openai_key)


@cl.on_message
async def main(message: str):
    # Use pipeline to get a response
    output = query_pipeline.run(query=message)

    # Create Chainlit message with response
    response = output['answers'][0].answer
    msg = cl.Message(content=response)

    # Send message to the user
    await msg.send()
