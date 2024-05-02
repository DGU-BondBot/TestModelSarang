#chat_models.py
from langchain_community.chat_models import ChatOllama

# LLM 모델 클래스 정의
class ChatOllamaModel:
    def __init__(self, model_name):
        self.llm = ChatOllama(model=model_name)

    def invoke(self, query):
        return self.llm.invoke(query)