from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient

class ChatbotClass:
    def __init__(self, model_name, device, llm_model, temperature, encode_kwargs, qdrant_url, collection_name):
        self.model_name = model_name
        self.device = device
        self.llm_model = llm_model
        self.temperature = temperature
        self.encode_kwargs = encode_kwargs
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs
        )
        self.prompt_template = '''
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Answer in the language of input message.

        Context: {context}
        Question: {question}
        
        Only return the helpful answer. Answer must be detailed and well explained.
        '''
        self.client = QdrantClient(url=self.qdrant_url, prefer_grpc=False)
        self.db = Qdrant(client=self.client, embedding_function=self.embeddings.embed_query, collection_name=self.collection_name)

        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=['context', 'question'])

        self.retriever = self.db.as_retriever(search_kwargs={"k": 1})

        self.qa = RetrievalQA.from_chain_type(
            llm=None,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            verbose=False
        )

    def get_response(self, query):
        try:
            return self.qa.run(query)
        except Exception as ex:
            return f"Error: {ex}"
