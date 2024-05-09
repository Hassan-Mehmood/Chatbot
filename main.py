from Chatbot import ChatSystem
from ChatInterface import ChatInterface

chat = ChatSystem()
chat_interface = ChatInterface(chat)
chat_interface.start_chat()