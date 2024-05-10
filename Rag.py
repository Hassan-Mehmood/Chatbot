from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class Rag:
	def __init__(self):
		self.splitter = RecursiveCharacterTextSplitter(
			separators=["\n\n", "\n", " "], 
			chunk_size=1000, 
			chunk_overlap=200
		)
		self.prompt = '''Answer the question only on the following context.
{context}
--------
Answer the question only on the above context. {question}
'''
		
		self.prompt_template = ChatPromptTemplate.from_messages([
			("system", "You will answer questions based on the context. And you will never answer any question without knowing the context. If you don't know the answer you will simply say that I don't know"),
			("human", self.prompt),
			])
		self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
		self.chain = self.prompt_template | self.llm

	def load_and_split_document(self, path):
		loader = PyPDFLoader(path)
		pages = loader.load_and_split(text_splitter=self.splitter)

		return pages

	def create_embeddings(self):
		embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
		return embeddings
	
	def create_vectorstore(self, pages, embeddings):
		vectorstore = FAISS.from_documents(pages, embeddings)
		return vectorstore
	
	def chat(self, vectorstore):
		while True:
			query = input("User: ")
			if query.lower() == 'q' or query.lower() == 'quit':
				break
			similar_docs = vectorstore.similarity_search(query, k=5)
			response = self.chain.invoke({
				"context": similar_docs,
				"question": query
			})

			print(f'''AI: {response.content}

Press 'q' or 'quit' to quit''')
		print('Goodbye!')
	