import os
import sys
import pinecone
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Replicate API token
os.environ['REPLICATE_API_TOKEN'] = "Replicate API"

# Initialize Pinecone
pinecone.init(api_key='Pinecone', environment='gcp-starter')

# Load and preprocess the PDF document
loader = PyPDFLoader('Substation-operation-and-maintenance-handbook.pdf')
documents = loader.load()

# Split the documents into smaller chunks for processing
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Use HuggingFace embeddings for transforming text into numerical vectors
embeddings = HuggingFaceEmbeddings()

# Set up the Pinecone vector database
index_name = 'sih'
index = pinecone.Index(index_name)
vectordb = Pinecone.from_existing_index(index_name,embeddings)
# Pinecone.from_documents(texts, embeddings, index_name=index_name)

# Initialize Replicate Llama2 Model
llm = Replicate(
    model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    input={"temperature": 0.75, "max_length": 3000}
)

# Set up the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True
)

if __name__ == "__main__":
    app.run(debug=True)
