#prompts.py
from langchain_core.prompts import ChatPromptTemplate

# 프롬프트 설정 클래스
class PromptTemplates:
    @staticmethod
    def get_basic_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "You are BondBot, an empathetic and wise counselor specializing in romantic relationships. Answer the question only in Korean"),
            ("user", "{input}")
        ])

    @staticmethod
    def get_translation_prompt():
        return ChatPromptTemplate.from_template("[{korean_input}] translate the question into English. Don't say anything else, just translate it.")