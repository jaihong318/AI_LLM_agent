import os
import tempfile
import json
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any

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
# Google Cloud Run 환경변수에 아래 값들을 반드시 설정해야 합니다.
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # 구글 AI API 키

# Supabase 클라이언트 초기화
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
except Exception as e:
    print(f"Supabase 초기화 에러: {e}")

# FastAPI 앱 초기화
app = FastAPI(title="RAG Chatbot API")

# CORS 설정 (Google Apps Script / 클라이언트 직접 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 임베딩 모델 (OpenAI)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# LLM 모델 (Google Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", 
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    streaming=True # 스트리밍 활성화
)

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] = [] # [{"role": "user", "content": "hello"}, ...]

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF 파일을 업로드받아 텍스트로 변환, 청크로 쪼갠 후 벡터 DB(Supabase)에 저장합니다."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    
    temp_file_path = ""
    try:
        # 1. 파일 임시 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # 2. PDF 읽기 및 텍스트 변환
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        if not documents:
            raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")

        # 3. 텍스트 청크 단위로 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # 4. Supabase Vector DB에 저장 (OpenAI 임베딩 사용)
        SupabaseVectorStore.from_documents(
            chunks,
            embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )

        return {"message": "RAG 생성이 완료되었습니다.", "chunks_count": len(chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 처리 중 오류 발생: {str(e)}")
    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """사용자의 질문과 과거 대화내역을 바탕으로 질의를 확장하고, RAG로 검색하여 답변을 스트리밍합니다."""
    
    # 1. 대화 기록 포맷팅 (LangChain Message 객체로 변환)
    formatted_history = []
    for msg in request.chat_history:
        if msg.get("role") == "user":
            formatted_history.append(HumanMessage(content=msg.get("content")))
        elif msg.get("role") == "assistant":
            formatted_history.append(AIMessage(content=msg.get("content")))

    # 2. 질의 확장 (Query Augmentation)
    # 시스템 메시지: 과거 대화를 기반으로 의도를 파악하여 한 문장의 명료한 질문으로 변환
    augment_prompt = ChatPromptTemplate.from_messages([
        ("system", "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])
    
    # 3. Query Augmentation Chain 구성
    augment_chain = augment_prompt | llm | StrOutputParser()
    
    # 4. 체인 실행 (데이터 주입)
    try:
        augmented_query = augment_chain.invoke({
            "chat_history": formatted_history,
            "query": request.query
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질의 확장 중 오류 발생: {str(e)}")

    # 5. 증강된 질의로 벡터 DB 검색 (Supabase Retriever)
    try:
        vectorstore = SupabaseVectorStore(
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        retrieved_docs = retriever.invoke(augmented_query)
        
        # 검색된 컨텍스트와 출처 정리
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        sources = [{"source": doc.metadata.get("source", "Unknown"), "page": doc.metadata.get("page", 0)} for doc in retrieved_docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문서 검색 중 오류 발생: {str(e)}")

    # 6. 최종 답변 생성을 위한 시스템 메시지 및 프롬프트 설계
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야. 주어진 Context만을 기반으로 사용자의 질문에 답해줘. 만약 Context에 답변이 없다면, 알 수 없다고 말해줘.\n\nContext: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{augmented_query}")
    ])

    answer_chain = answer_prompt | llm | StrOutputParser()

    # 7. 실시간 스트리밍 핸들러 (Generator) 생성
    async def generate_response():
        # 먼저 클라이언트 측 화면 분리를 위해 검색된 Context와 출처를 JSON 문자열로 전송합니다.
        # 구분자를 통해 클라이언트가 파싱할 수 있게 함
        meta_data = json.dumps({"type": "sources", "sources": sources, "context": context, "augmented_query": augmented_query})
        yield f"data: {meta_data}\n\n"
        
        # LLM의 답변을 스트리밍
        try:
            async for chunk in answer_chain.astream({
                "context": context,
                "chat_history": formatted_history,
                "augmented_query": augmented_query
            }):
                # SSE 포맷 규격에 맞춰 전송
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            
            yield "data: {\"type\": \"done\"}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': f'LLM 답변 생성 에러: {str(e)}'})}\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")
