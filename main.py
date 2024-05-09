from ChatSystem import ChatSystem
from ChatInterface import ChatInterface
from Rag import Rag

# chat = ChatSystem()
# chat_interface = ChatInterface(chat)
# chat_interface.start_chat()

rag = Rag()

pages = rag.load_and_split_document("data/rag.pdf")
embeddings = rag.create_embeddings()
vectorstore = rag.create_vectorstore(pages, embeddings)

docs = vectorstore.similarity_search("What are the rulings about a fair trail", k=1)
for doc in docs:
    print(doc.page_content)
