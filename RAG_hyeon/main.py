"""
[현진건 작가 페르소나 RAG 서버 - Google Cloud Run 배포용]
- 임베딩 모델: models/text-embedding-004
- 생성 모델: gemini-1.5-flash / gemini-1.5-pro
- 벡터 DB: Chroma DB (/tmp/chroma_storage)
- 검색기: LangChain MultiQueryRetriever (다중 질문 질의 확장)
- 체인 구성: LangChain LCEL (RunnablePassthrough, ChatPromptTemplate)
- 스트리밍: Server-Sent Events (SSE) 실시간 스트리밍
"""

import os
import json
import logging
import tempfile
import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# LangChain 및 Google Generative AI 컴포넌트 임포트
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 로깅 설정: 디버깅 및 Cloud Run 로깅에 최적화
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("HyunJinGeonBot")

# Google API Key 검증 (Cloud Run 환경변수에서 로드)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("경고: GOOGLE_API_KEY 환경변수가 설정되지 않았습니다. API 호출 시 오류가 발생할 수 있습니다.")

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="현진건 작가 페르소나 RAG API",
    description="소설 '운수 좋은 날'의 현진건 작가 문체로 질의에 답하는 RAG 시스템",
    version="1.0.0"
)

# CORS 미들웨어 등록 (Google Apps Script Web App 및 브라우저 직접 호출 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ChromaDB 영구 저장 디렉토리 (Cloud Run의 쓰기 가능 영역은 /tmp)
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIR", "/tmp/chroma_db")

# 전역 임베딩 인스턴스 초기화 (models/text-embedding-004 지정)
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    logger.info("Google text-embedding-004 모델 초기화 완료.")
except Exception as e:
    logger.error(f"임베딩 모델 초기화 실패: {str(e)}")
    embeddings = None

# Chroma 벡터 스토어 초기화
vector_store = Chroma(
    collection_name="hyunjingeon_docs",
    embedding_function=embeddings,
    persist_directory=CHROMA_PERSIST_DIRECTORY
)

# 현진건 작가 페르소나 프롬프트 템플릿
# 프롬프트 허브(LangChain Hub)의 RAG 템플릿 구조를 기반으로 현진건 작가의 현실주의 문체 페르소나 주입
SYSTEM_PROMPT = """당신은 소설 <운수 좋은 날>, <빈처>, <B사감과 러브레터>를 집필한 한국 근대 리얼리즘 문학의 선구자, 현진건 작가입니다.

[답변 지침]
1. 문체: 1920년대 식민지 조선의 암담한 현실과 민중의 비애를 꿰뚫어 보는 근대 한국 문학의 현실주의(리얼리즘) 어조(~하오, ~소, ~했단 말이지, ~인 게지, 설렁탕이라도 사다 줬으면 좋으련만 등의 특유의 서글픈 페이소스와 담담하면서도 날카로운 문체)로 사용자의 질문에 답하십시오.
2. 근거 기반: 반드시 아래에 주어지는 [참고 자료]의 내용을 바탕으로 성실하고 정확하게 사실관계를 설명하되, 작가로서의 통찰과 문학적 표현을 곁들이십시오.
3. 정직성: [참고 자료]에 질문에 대한 내용이 없다면, "내 눈으로 보지 못한 것을 억지로 꾸며낼 재주는 없소이다. 주어진 글귀에는 그와 관련된 사연이 적혀있지 않구려..."라며 완곡하고 솔직하게 밝히십시오.

[참고 자료]
{context}

사용자의 질문: {question}
현진건 작가의 답변:"""

prompt_template = ChatPromptTemplate.from_template(SYSTEM_PROMPT)


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
async def health_check():
    """서버 가동 상태 및 ChromaDB 청크 수 확인 엔드포인트"""
    try:
        count = vector_store._collection.count()
        return {
            "status": "healthy",
            "message": "현진건 봇 서버가 정상 가동 중입니다.",
            "indexed_chunks_count": count,
            "embedding_model": "models/text-embedding-004"
        }
    except Exception as e:
        logger.error(f"헬스체크 중 예외 발생: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": f"상태 진단 오류: {str(e)}"}
        )


@app.post("/index-pdf")
async def index_pdf(file: UploadFile = File(...)):
    """
    업로드된 PDF 문서를 텍스트로 추출하고 청크로 분할하여 ChromaDB에 저장하는 엔드포인트
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="PDF 형식의 파일만 처리할 수 있습니다. (확장자 .pdf 확인 요망)"
        )

    temp_path = None
    try:
        logger.info(f"PDF 파일 수신 시작: {file.filename}")
        # 임시 디렉토리에 업로드된 PDF 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name

        # 1. PyPDFLoader로 PDF 텍스트 추출
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        if not documents:
            raise HTTPException(
                status_code=422,
                detail="PDF 파일에서 텍스트를 추출할 수 없습니다. 스캔 이미지 전용 PDF인지 확인해 주세요."
            )

        logger.info(f"PDF 파싱 완료. 총 {len(documents)}페이지 추출됨.")

        # 2. RecursiveCharacterTextSplitter로 청크 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"청크 분할 완료. 총 {len(split_docs)}개 청크 생성됨.")

        # 각 청크 메타데이터에 원본 파일명 및 순번 보강
        for i, doc in enumerate(split_docs):
            doc.metadata["source_filename"] = file.filename
            doc.metadata["chunk_index"] = i

        # 3. Chroma DB에 임베딩 벡터 생성 및 인덱싱
        vector_store.add_documents(split_docs)
        total_count = vector_store._collection.count()
        logger.info(f"ChromaDB 인덱싱 성공! 현재 총 청크 수: {total_count}")

        return {
            "success": True,
            "filename": file.filename,
            "total_pages": len(documents),
            "chunks_created": len(split_docs),
            "database_total_chunks": total_count,
            "message": f"'{file.filename}' 문서가 성공적으로 색인되어 현진건 작가의 서재에 편입되었소."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("PDF 인덱싱 중 치명적 오류 발생")
        raise HTTPException(
            status_code=500,
            detail=f"서버 내부 색인 작업 실패: {str(e)}"
        )
    finally:
        # 임시 파일 정리
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/stream-query")
async def stream_query(request: QueryRequest):
    """
    다중 질문 검색기(MultiQueryRetriever)와 LCEL(RunnablePassthrough),
    실시간 토큰 스트리밍(SSE)을 적용한 질의응답 엔드포인트
    """
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="질문 내용이 비어 있습니다. 문장이나 단어를 입력해 주십시오.")

    # 저장된 문서 수 확인
    total_chunks = vector_store._collection.count()
    if total_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail="참고할 서책(PDF)이 아직 인덱싱되지 않았소. 먼저 PDF 파일을 업로드하여 색인을 생성해 주시오."
        )

    # 1. 다중 질의 확장을 위한 LLM 인스턴스 (다양한 관점의 검색어 생성)
    query_expansion_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY
    )

    # 2. 기본 Chroma 검색기 구성
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # 3. LangChain MultiQueryRetriever 구성 (사용자 질문을 다각도로 확장하여 검색 누락 방지)
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=query_expansion_llm
    )

    # 4. 실시간 응답 생성을 위한 Gemini 스트리밍 LLM (현실주의 문체 생성)
    generation_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.6,
        streaming=True,
        google_api_key=GOOGLE_API_KEY
    )

    # 문서 포맷팅 헬퍼 함수
    def format_docs(docs):
        formatted = []
        for i, d in enumerate(docs):
            page = d.metadata.get("page", 0) + 1
            src = d.metadata.get("source_filename", "참고문서")
            formatted.append(f"[발췌문 #{i+1} | 출처: {src} (p.{page})]\n{d.page_content}")
        return "\n\n".join(formatted)

    # SSE 제너레이터 함수 정의
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            logger.info(f"다중 질의 검색 시작: '{user_query}'")
            # 비동기로 MultiQueryRetriever를 통해 문서 검색 수행
            retrieved_docs = await asyncio.to_thread(multi_query_retriever.invoke, user_query)

            # 클라이언트에 먼저 검색된 청크(출처 및 원문)를 이벤트로 전송
            sources_payload = [
                {
                    "source": doc.metadata.get("source_filename", "문서"),
                    "page": doc.metadata.get("page", 0) + 1,
                    "content": doc.page_content.strip()
                }
                for doc in retrieved_docs
            ]

            yield f"event: sources\ndata: {json.dumps(sources_payload, ensure_ascii=False)}\n\n"

            # LCEL 파이프라인 구성 (RunnablePassthrough 활용)
            context_text = format_docs(retrieved_docs)
            chain = (
                {
                    "context": lambda x: context_text,
                    "question": RunnablePassthrough()
                }
                | prompt_template
                | generation_llm
                | StrOutputParser()
            )

            # 토큰 단위 비동기 스트리밍 순회
            async for chunk in chain.astream(user_query):
                # SSE 데이터 형식 규격 준수 (줄바꿈 문자는 이중 이스케이프 방지를 위해 json 직렬화)
                token_data = json.dumps({"token": chunk}, ensure_ascii=False)
                yield f"event: token\ndata: {token_data}\n\n"

            # 완료 신호 전송
            yield "event: done\ndata: {}\n\n"
            logger.info("답변 스트리밍 정상 완료.")

        except Exception as err:
            logger.exception("답변 생성 중 스트리밍 에러 발생")
            err_data = json.dumps({"error": f"작가 응답 생성 실패: {str(err)}"}, ensure_ascii=False)
            yield f"event: error\ndata: {err_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Nginx/Proxy 버퍼링 방지
        }
    )


if __name__ == "__main__":
    import uvicorn
    # 로컬 개발 및 컨테이너 테스트용 포트 지정 (Cloud Run 기본 PORT 환경변수 준수)
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
