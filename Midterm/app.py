```python
import os
import shutil
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


app = FastAPI(title="Seoul Urban Planner RAG API")


# =========================================================
# 1. CORS 설정
# GAS 및 외부 클라이언트 접근 허용
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 2. RAG 설정
# Google 공식 임베딩 모델로 변경
# =========================================================

RAG_CONFIG = {
    "vector_db": "Chroma DB",
    "embedding_model": "models/gemini-embedding-001",
    "llm_model": "gemini-3.8-flash",
    "multi_query_retriever": "MultiQueryRetriever (다중 질문 검색기)",
    "query_expansion": "LCEL RunnablePassthrough 기반 질의 확장",
    "system_role": "당신은 도시 계획 설계자입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요."
}


# =========================================================
# 3. 환경 변수 및 경로 설정
# =========================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

CHROMA_PERSIST_DIR = "/tmp/chroma_db"


# =========================================================
# 4. Google AI 모델 초기화
# =========================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model=RAG_CONFIG["embedding_model"],
    google_api_key=GOOGLE_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model=RAG_CONFIG["llm_model"],
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)


# =========================================================
# 5. 전역 변수
# =========================================================

vectorstore: Optional[Chroma] = None
rag_chain = None


# =========================================================
# 6. 문서 포맷팅 함수
# =========================================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# =========================================================
# 7. RAG 체인 초기화
# MultiQueryRetriever + LCEL
# =========================================================

def init_chain(vs: Chroma):
    global rag_chain

    # 기본 검색기 구성
    base_retriever = vs.as_retriever(
        search_kwargs={"k": 3}
    )

    # 다중 질문 검색기 구성
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    # 서울시 특화 도시 계획가 시스템 프롬프트
    system_prompt = (
        f"{RAG_CONFIG['system_role']}\n\n"
        "제공된 서울시 도시계획 가이드 문서를 바탕으로 "
        "구체적이고 전문적으로 답변하세요.\n\n"
        "문서에 근거가 없는 내용은 사실처럼 단정하지 마세요.\n"
        "문서 내용을 바탕으로 답변할 수 없는 경우 "
        "해당 정보가 문서에 없다고 안내하세요.\n\n"
        "[도시계획 문서 컨텍스트]\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    # LCEL 파이프라인 구성
    rag_chain = (
        {
            "context": multi_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )


# =========================================================
# 8. 요청 데이터 모델
# =========================================================

class QueryRequest(BaseModel):
    question: str


# =========================================================
# 9. Health Check
# =========================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Seoul Urban Planner RAG"
    }


# =========================================================
# 10. RAG 설정 확인
# =========================================================

@app.get("/config")
def get_config():
    return RAG_CONFIG


# =========================================================
# 11. Google AI 지원 모델 확인
# =========================================================

@app.get("/models")
def list_supported_models():
    try:
        genai.configure(api_key=GOOGLE_API_KEY)

        embed_models = [
            m.name
            for m in genai.list_models()
            if "embedContent" in m.supported_generation_methods
        ]

        return {
            "status": "success",
            "configured_embedding_model": RAG_CONFIG["embedding_model"],
            "available_embedding_models": embed_models
        }

    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }


# =========================================================
# 12. PDF 업로드 및 벡터 DB 색인
# =========================================================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    # PDF 파일 형식 확인
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="PDF 파일만 업로드할 수 있습니다."
        )

    # 임시 파일명 생성
    temp_filename = f"{uuid.uuid4().hex}.pdf"
    temp_path = os.path.join("/tmp", temp_filename)

    try:
        # PDF 임시 저장
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # PDF 문서 로드
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs:
            raise HTTPException(
                status_code=400,
                detail="PDF에서 문서 내용을 추출하지 못했습니다."
            )

        # 문서 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="PDF에서 색인할 텍스트를 찾지 못했습니다."
            )

        # 기존 컬렉션 초기화 후 새 임베딩으로 색인
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name="seoul_urban_planner"
        )

        # RAG 체인 초기화
        init_chain(vectorstore)

        return {
            "status": "success",
            "embedding_model": RAG_CONFIG["embedding_model"],
            "message": (
                f"총 {len(docs)}페이지, "
                f"{len(chunks)}개 청크가 "
                "Chroma DB에 색인되었습니다."
            )
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF 업로드 및 색인 실패: {str(e)}"
        )

    finally:
        # 임시 PDF 파일 삭제
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =========================================================
# 13. RAG 기반 질문 및 답변
# =========================================================

@app.post("/chat")
def chat(req: QueryRequest):
    global rag_chain

    if rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="PDF 문서를 먼저 업로드하여 임베딩을 완료해주세요."
        )

    try:
        response = rag_chain.invoke(req.question)

        return {
            "question": req.question,
            "answer": response
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        )
```
