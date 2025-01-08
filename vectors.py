from langchain_community.document_loaders import PyPDFLoader
from langchain_community import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import os

class Embeddings:
    def __init__(self, model_name, device, encode_kwargs, qdrant_url, connection_name):
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.qdrant_url = qdrant_url
        self.connection_name = connection_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs
        )

    def create_embeddings(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError('The file path does not exist.')
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
        splits = text_splitter.split_documents(docs)
        try:
            Qdrant.from_documents(
                splits,
                self.embeddings,
                url=self.qdrant_url,
                collection_name=self.connection_name
            )
        except ConnectionError as ce:
            raise ConnectionError("Failed to connect to Qdrant.")
        return "Qdrant vector database is created!"
