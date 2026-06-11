# pip install -U langchain-community langchain-text-splitters pypdf langchain-openai langchain-chroma chromadb

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def input_positive_int(prompt, default):
    value = input(f"{prompt} (기본값: {default}): ").strip()
    if not value:
        return default

    try:
        number = int(value)
    except ValueError:
        print(f"숫자가 아니므로 기본값 {default}을 사용합니다.")
        return default

    if number <= 0:
        print(f"0보다 큰 값을 입력해야 하므로 기본값 {default}을 사용합니다.")
        return default

    return number


def create_vector_db(documents, persist_directory):
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되어 있지 않아 임베딩/벡터DB 저장을 건너뜁니다.")
        return None

    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        print("임베딩/벡터DB 패키지가 설치되어 있지 않습니다.")
        print("다음 명령으로 설치해 주세요: pip install -U langchain-openai langchain-chroma chromadb")
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )

    if hasattr(vector_store, "persist"):
        vector_store.persist()

    print(f"벡터DB 저장을 완료했습니다: {persist_directory}")
    return vector_store


def answer_question(vector_store, question):
    from langchain_openai import ChatOpenAI

    related_docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join(
        f"[문서 {index}]\n{doc.page_content}"
        for index, doc in enumerate(related_docs, start=1)
    )

    prompt = f"""
다음 PDF 문서 내용만 참고해서 사용자의 질문에 답하세요.
문서 내용에 없는 정보는 모른다고 답하세요.

[PDF 문서 내용]
{context}

[사용자 질문]
{question}

[답변]
"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(prompt)
    return response.content


def start_question_screen(vector_store):
    if vector_store is None:
        print("질문 답변 화면을 시작할 수 없습니다. API 키와 임베딩/벡터DB 설정을 확인해 주세요.")
        return

    print("\n==============================")
    print("PDF 질문 답변 화면")
    print("질문을 입력하면 PDF 내용을 바탕으로 답변합니다.")
    print("종료하려면 exit 또는 종료를 입력하세요.")
    print("==============================")

    while True:
        question = input("\n질문: ").strip()
        if question.lower() in ("exit", "quit", "q") or question == "종료":
            print("질문 답변 화면을 종료합니다.")
            break

        if not question:
            print("질문을 입력해 주세요.")
            continue

        answer = answer_question(vector_store, question)
        print("\n답변:")
        print(answer)


chunk_size = input_positive_int("chunk 크기를 입력하세요", 300)
chunk_overlap = input_positive_int("chunk overlap 값을 입력하세요", 20)

if chunk_overlap >= chunk_size:
    print("chunk overlap은 chunk 크기보다 작아야 하므로 기본값 20을 사용합니다.")
    chunk_overlap = 20

# PDF 로더 인스턴스 생성
pdf_path = Path(__file__).parent / "luck.pdf"
loader = PyPDFLoader(str(pdf_path))
# PDF 파일에서 페이지를 로드하고 분할하여 페이지 객체 리스트로 변환
pages = loader.load_and_split()

# Split 단계(텍스트 청크 쪼개기)
# LLM이 처리하기 좋도록 문서를 작은 단위(chunk)로 나눔
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,  # 하나의 텍스트 조각에 들어갈 최대 글자 수
    chunk_overlap=chunk_overlap,  # 앞뒤 텍스트 조각 사이에 겹칠 글자 수. 문맥 끊김을 줄이기 위해 보통 10~20% 정도 겹치게 설정
    length_function=len,
    is_separator_regex=False,  # 구분자를 정규표현식으로 해석할지 여부
)

# 설정한 chunk_size와 chunk_overlap에 따라 페이지 객체 리스트를 텍스트 조각 리스트로 변환
texts = text_splitter.split_documents(pages)
print("............" + str(len(texts)) + "개의 텍스트 조각이 생성되었습니다............")

if texts:
    output_path = Path(__file__).parent / "chunks.txt"
    with output_path.open("w", encoding="utf-8") as output_file:
        for index, text in enumerate(texts, start=1):
            output_file.write(f"--- [Chunk {index}] ---\n")
            output_file.write(text.page_content)
            output_file.write("\n\n")

    print(f"모든 텍스트 조각을 저장했습니다: {output_path}")

    vector_db_path = Path(__file__).parent / "chroma_db"
    vector_store = create_vector_db(texts, vector_db_path)

    print("--- [첫 번째 텍스트 조각(Chunk) 객체 출력] ---")
    print(texts[0])

    print("\n--- [첫 번째 조각의 실제 텍스트 내용만 출력] ---")
    print(texts[0].page_content)

    start_question_screen(vector_store)
else:
    print("분할된 텍스트 조각이 없습니다. PDF 파일 내용을 확인해 주세요.")
