import dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()

class Rag:
	def __init__(self):
		self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
		self.splitter = RecursiveCharacterTextSplitter(
			separators=["\n\n", "\n", " "], 
			chunk_size=1000, 
			chunk_overlap=200
		)	

	def load_and_split_document(self, path):
		loader = PyPDFLoader(path)
		pages = loader.load_and_split(text_splitter=self.splitter)

		return pages

	def create_embeddings(self):
		embeddings = OpenAIEmbeddings()
		return embeddings
	
	def create_vectorstore(self, pages, embeddings):
		vectorstore = FAISS.from_documents(pages, embeddings)
		return vectorstore
	
	