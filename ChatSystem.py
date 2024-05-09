import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory

dotenv.load_dotenv()

class ChatSystem:
    def __init__(self, model="gpt-3.5-turbo-1106", temperature=0.2):
        self.chat = ChatOpenAI(model=model, temperature=temperature)
        self.chat_history = ChatMessageHistory()
        self.chain = self.get_chain()

    def get_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Answer all questions to the best of your ability."), 
                MessagesPlaceholder(variable_name="messages"),
            ])
        return prompt | self.chat

    def get_answer(self):
        response = self.chain.invoke({
            "messages": self.chat_history.messages,
        })
        return response