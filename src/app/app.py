import chainlit as cl
from faiss_doc_store import initialize_documents, initialize_faiss_document_store, initialize_rag_pipeline
import os


# Load environment variables
openai_key = os.environ['OPENAI_API_KEY']


# Initialize documents
filepath = 'data/recipes_prepared_100.csv'
if not os.path.isfile(filepath):        
       print("Data file data/recipes_prepared_100.csv not found")
       
documents = initialize_documents(filepath)

# Initialize document store and retriever
document_store, retriever = initialize_faiss_document_store(documents=documents)

# Initialize pipeline
query_pipeline = initialize_rag_pipeline(retriever=retriever, openai_key=openai_key)


@cl.on_message
async def main(message: str):
    # Use pipeline to get a response
    output = query_pipeline.run(query=message)

    # Create Chainlit message with response
    response = output['answers'][0].answer
    msg = cl.Message(content=response)

    # Send message to the user
    await msg.send()
