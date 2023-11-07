from dotenv import dotenv_values
from langchain.memory import MongoDBChatMessageHistory

config = dotenv_values(".env")


class Store:
    def __init__(self, sessionId="test1"):
        self.sessionId = sessionId
        self.message_history = MongoDBChatMessageHistory(
            connection_string=config["MONGO_URI"],
            session_id=sessionId,
        )

    def update_history(self, question, answer):
        self.message_history.add_user_message(question)
        self.message_history.add_ai_message(answer)


if __name__ == "__main__":
    pass
