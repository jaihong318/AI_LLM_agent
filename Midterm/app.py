import os
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="Seoul Urban Planner RAG API")

# GAS 및 외부 접근 허용을 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 과제 요구 명세 메타데이터
RAG_CONFIG = {
    "vector_db": "Chroma DB",
    "embedding_model": "models/text-embedding-004",
    "llm_model": "gemini-1.5-flash",
    "multi_query_retriever": "MultiQueryRetriever (다중 질문 검색기)",
    "query_expansion": "LCEL RunnablePassthrough 기반 질의 확장",
    "system_role": "당신은 도시 계획 설계자입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요."
}

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
CHROMA_PERSIST_DIR = "/tmp/chroma_db"

embeddings = GoogleGenerativeAIEmbeddings(
    model=RAG_CONFIG["embedding_model"],
    google_api_key=GOOGLE_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model=RAG_CONFIG["llm_model"],
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

vectorstore: Optional[Chroma] = None
rag_chain = None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def init_chain(vs: Chroma):
    global rag_chain
    base_retriever = vs.as_retriever(search_kwargs={"k": 3})
    
    # 1. 다중 질문 검색기(MultiQueryRetriever) 구성
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    # 2. 서울시 특화 도시 계획가 시스템 프롬프트 주입
    system_prompt = (
        f"{RAG_CONFIG['system_role']}\n\n"
        "제공된 서울시 도시계획 가이드 문서를 바탕으로 구체적이고 전문적으로 답변하세요.\n\n"
        "[도시계획 문서 컨텍스트]\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    # 3. RunnablePassthrough를 적용한 질의 전달 및 LCEL 파이프라인
    rag_chain = (
        {
            "context": multi_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

class QueryRequest(BaseModel):
    question: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Seoul Urban Planner RAG"}

@app.get("/config")
def get_config():
    return RAG_CONFIG

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        init_chain(vectorstore)
        
        return {
            "status": "success",
            "message": f"총 {len(docs)}페이지, {len(chunks)}개 청크가 Chroma DB에 색인되었습니다."
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/chat")
def chat(req: QueryRequest):
    global rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=400, detail="PDF 문서를 먼저 업로드하여 임베딩을 완료해주세요.")
    
    response = rag_chain.invoke(req.question)
    return {"question": req.question, "answer": response}
