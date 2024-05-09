class ChatInterface:
    def __init__(self, chat):
        self.chat = chat

    def start_chat(self):
        while True:
            message = input("User: ")
            if message.lower() == 'q' or message.lower() == 'quit':
                break

            self.chat.chat_history.add_user_message(message)

            response = self.chat.get_answer()

            self.chat.chat_history.add_ai_message(response)

            print(f'''
AI: {response.content}

Press 'q' or 'quit' to quit''')