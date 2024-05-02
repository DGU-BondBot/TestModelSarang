#output_parsers.py
from langchain_core.output_parsers import StrOutputParser

# 출력 파서 클래스 정의
class SimpleStrOutputParser:
    @staticmethod
    def parse(output):
        return StrOutputParser().parse(output)