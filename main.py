import dotenv
from ChatSystem import ChatSystem
from ChatInterface import ChatInterface
from Rag import Rag


dotenv.load_dotenv()
# chat = ChatSystem()
# chat_interface = ChatInterface(chat)
# chat_interface.start_chat()

rag = Rag()

print('Welcome to the Langchain-Rag demo!')

print('Loading documents...')
pages = rag.load_and_split_document("data/ml_book.pdf")
print('Creating embeddings...')
embeddings = rag.create_embeddings()
print('Creating vector store...')
vectorstore = rag.create_vectorstore(pages, embeddings)

rag.chat(vectorstore)

