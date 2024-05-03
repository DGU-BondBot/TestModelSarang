from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from prompts import PromptTemplates
from chat_models import ChatOllamaModel
from langchain_core.runnables import RunnablePassthrough
from output_parsers import SimpleStrOutputParser

loader = TextLoader("./conversationData.txt", encoding="utf-8")
docs = loader.load()
# print(f"문서의 수: {len(docs)}")

# 10번째 페이지의 내용 출력
# print(f"\n[페이지내용]\n{docs[0].page_content[:500]}")
# print(f"\n[metadata]\n{docs[0].metadata}\n")    

# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

splits = text_splitter.split_documents(docs)

# 단계 3: 임베딩 & 벡터스토어 생성(Create Vectorstore)
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(
    documents=splits, embedding=HuggingFaceBgeEmbeddings()
)

# 단계 4: 검색(Search)
# 뉴스에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# 단계 5: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = PromptTemplates.get_basic_prompt()

# 단계 6: 언어모델 생성(Create LLM)
# 모델(LLM) 을 생성합니다.
llm = ChatOllamaModel("llama3:latest")

output_parser = SimpleStrOutputParser()

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


# 단계 7: 체인 생성(Create Chain)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser.parse
)

# 단계 8: 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
question = "저는 ENFP 유형이고 남자친구는 ISTJ예요. 요즘 우리 관계가 좀 어색한 것 같아요. 뭔가 서로 잘 통하지 않는 느낌이 드는데, 어떻게 하면 좋을까요?"
response = rag_chain.invoke(question)

# 결과 출력
print(f"[HUMAN]\n{question}\n")
print(f"[AI]\n{response}")
