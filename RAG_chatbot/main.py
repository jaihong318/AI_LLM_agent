import os
import tempfile
import json
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- 환경 변수 로드 ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
# 다양한 키 이름(SERVICE_KEY, SERVICE_ROLE_KEY, KEY)을 모두 허용하여 불일치 방지
SUPABASE_SERVICE_KEY = (
    os.environ.get("SUPABASE_SERVICE_KEY")
    or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    or os.environ.get("SUPABASE_KEY")
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Supabase 클라이언트 전역 변수 초기화 (기본값 None으로 선언하여 NameError 방지)
supabase: Optional[Client] = None

# Supabase 클라이언트 연결 시도
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print(f"[초기화 경고] Supabase 환경 변수가 누락되었습니다. URL: {bool(SUPABASE_URL)}, KEY: {bool(SUPABASE_SERVICE_KEY)}")
else:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("Supabase 클라이언트가 성공적으로 초기화되었습니다.")
    except Exception as e:
        print(f"Supabase 클라이언트 초기화 실패: {str(e)}")

# FastAPI 앱 초기화
app = FastAPI(title="도시 정책 전문가 AI - RAG API")

# CORS 설정 (Google Apps Script / 웹 클라이언트 직접 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 임베딩 모델 (OpenAI)
embeddings = None
if OPENAI_API_KEY:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
else:
    print("[경고] OPENAI_API_KEY 환경 변수가 등록되지 않았습니다.")

# LLM 모델 (Google Gemini)
llm = None
if GOOGLE_API_KEY:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        streaming=True
    )
else:
    print("[경고] GOOGLE_API_KEY 환경 변수가 등록되지 않았습니다.")


class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] = []  # [{"role": "user", "content": "..."}, ...]


@app.get("/")
def health_check():
    """서버 상태 확인용 기본 엔드포인트"""
    return {
        "status": "healthy",
        "supabase_connected": supabase is not None,
        "embeddings_ready": embeddings is not None,
        "llm_ready": llm is not None
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF 파일을 업로드받아 텍스트로 변환, 청크 분할 후 Supabase 벡터 DB에 저장합니다."""
    # 1. 필수 의존성 및 환경 변수 검증
    if supabase is None:
        raise HTTPException(
            status_code=500,
            detail="서버에 Supabase 클라이언트가 초기화되지 않았습니다. Cloud Run 환경 변수(SUPABASE_URL, SUPABASE_SERVICE_KEY)를 확인하세요."
        )
    if embeddings is None:
        raise HTTPException(
            status_code=500,
            detail="OpenAI 임베딩 모델이 초기화되지 않았습니다. Cloud Run 환경 변수(OPENAI_API_KEY)를 확인하세요."
        )
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 확장자를 가진 파일만 업로드할 수 있습니다.")

    temp_file_path = ""
    try:
        # 2. 임시 파일로 PDF 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # 3. PDF 읽기 및 텍스트 로드
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        if not documents:
            raise HTTPException(status_code=400, detail="선택한 PDF에서 추출할 수 있는 텍스트가 없습니다.")

        # 4. 청크 분할 (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # 각 청크 메타데이터에 원본 파일명 반영
        for chunk in chunks:
            chunk.metadata["source"] = file.filename

        # 5. Supabase Vector DB에 저장
        SupabaseVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )

        return {
            "message": "RAG 벡터 DB 생성이 성공적으로 완료되었습니다.",
            "filename": file.filename,
            "chunks_count": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 처리 및 벡터 저장 중 오류 발생: {str(e)}")
    finally:
        # 임시 파일 정리
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """대화 이력 기반 질의 확장(Query Augmentation) 후 RAG 검색 및 실시간 SSE 스트리밍 답변을 제공합니다."""
    # 필수 구성 요소 검증
    if supabase is None or embeddings is None:
        raise HTTPException(status_code=500, detail="Supabase 또는 임베딩 모델 설정이 올바르지 않습니다.")
    if llm is None:
        raise HTTPException(status_code=500, detail="Gemini LLM이 초기화되지 않았습니다. GOOGLE_API_KEY를 확인하세요.")

    # 1. 대화 기록(Chat History) 포맷팅
    formatted_history = []
    for msg in request.chat_history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            formatted_history.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted_history.append(AIMessage(content=content))

    # 2. 질의 확장 (Query Augmentation Chain)
    augment_prompt = ChatPromptTemplate.from_messages([
        ("system", "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])
    augment_chain = augment_prompt | llm | StrOutputParser()

    try:
        augmented_query = await augment_chain.ainvoke({
            "chat_history": formatted_history,
            "query": request.query
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질의 확장(Query Augmentation) 실패: {str(e)}")

    # 3. Supabase 벡터 검색 수행
    try:
        vectorstore = SupabaseVectorStore(
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        retrieved_docs = retriever.invoke(augmented_query)

        # 컨텍스트 본문 및 출처 목록 구성
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        sources = [
            {
                "source": doc.metadata.get("source", "알 수 없음"),
                "page": doc.metadata.get("page", 0)
            }
            for doc in retrieved_docs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase 문서 검색(Retriever) 실패: {str(e)}")

    # 4. 도시 정책 전문가 프롬프트 구성
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야. 주어진 Context만을 기반으로 사용자의 질문에 답해줘. 만약 Context에 답변이 없다면, 알 수 없다고 말해줘.\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{augmented_query}")
    ])
    answer_chain = answer_prompt | llm | StrOutputParser()

    # 5. SSE(Server-Sent Events) 스트리밍 생성기
    async def generate_response():
        # 첫 번째 패킷: 확장된 질문과 검색된 출처 및 원문 컨텍스트 전달
        meta_data = json.dumps({
            "type": "sources",
            "augmented_query": augmented_query,
            "sources": sources,
            "context": context
        }, ensure_ascii=False)
        yield f"data: {meta_data}\n\n"

        # 두 번째 이후 패킷: 모델 토큰 실시간 스트리밍
        try:
            async for chunk in answer_chain.astream({
                "context": context,
                "chat_history": formatted_history,
                "augmented_query": augmented_query
            }):
                yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': f'LLM 답변 생성 에러: {str(e)}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")
