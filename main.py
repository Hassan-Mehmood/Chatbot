import dotenv
from ChatSystem import ChatSystem
from ChatInterface import ChatInterface
from Rag import Rag


dotenv.load_dotenv()
# chat = ChatSystem()
# chat_interface = ChatInterface(chat)
# chat_interface.start_chat()

rag = Rag()

pages = rag.load_and_split_document("data/ml_book.pdf")
embeddings = rag.create_embeddings()
vectorstore = rag.create_vectorstore(pages, embeddings)

docs = vectorstore.similarity_search("What is regression analysis", k=1)
print(f'DOCS: {docs} \n\n')
for doc in docs:
    print(f'Document page content: {doc.page_content}\n\n')
