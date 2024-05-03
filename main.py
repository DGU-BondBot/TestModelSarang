#main.py
from chat_models import ChatOllamaModel
from prompts import PromptTemplates
from output_parsers import SimpleStrOutputParser

def main():
    # LLM 모델 및 프롬프트, 출력 파서 초기화
    llm = ChatOllamaModel("llama3:latest")
    prompt = PromptTemplates.get_basic_prompt()
    output_parser = SimpleStrOutputParser()

    # 무한 루프 시작
    while True:
        # 사용자의 입력을 받음
        user_input = input("연애 고민을 입력하세요 (or type 'exit' to quit): ")

        # 사용자가 "exit"을 입력하면 루프를 종료
        if user_input.lower() == "exit":
            print("종료합니다.")
            break

        # 연쇄 작업 생성
        chain = prompt | llm.llm | output_parser.parse

        # 사용자의 입력을 처리
        my_input = {"input": user_input}
        response = chain.invoke(my_input)

        # 결과 출력
        print(f'Processed Message: {response.content}')

if __name__ == "__main__":
    main()
