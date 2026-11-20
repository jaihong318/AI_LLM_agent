"""FastAPI + LangGraph AI Journalist Server for graph_reporter."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

SYSTEM_PROMPT_TEMPLATE = """너는 신문기자이다.
최근 {about}에 대해 비판하는 심층 분석 기사를 쓰려고 한다.

단계별 기사 작성 지시사항
- 최근 어떤 이슈가 있는지 검색하고 사람들이 제일 관심있어 할만한 주체를 선정하고 왜 선정했는지 말해줘
- 그 내용으로 원고를 작성하기 위한 목차를 만들고 목차 내용을 채우기 위해 추가로 검색할 내용을 리스트로 정리해봐
- 검색할 리스트를 토대로 재검색해
- 목차에 있는 내용을 작성하기 위해 더 검색이 필요한 정보가 있는지 확인하고 있다면 추가로 검색해
- 검색된 결과에서 원하는 정보를 찾지 못했다면 다른 검색어로 재검색해도 좋아.

최종 기사 출력 조건
더이상 검색할 내용이 없다면 신문 기사 형식으로 최종 기사를 작성한다.
제목, 부제, 리드문, 본문의 구성으로 작성하며, 본문내용은 심층분석 기사에 맞게 구체적이고 깊이있게 작성해야 한다.

반드시 최종 기사를 작성할 때는 아래 마크다운 구분 헤더를 포함하여 출력하라:
# 제목: <기사 제목>
## 부제: <기사 부제>
### 리드문
<리드문 내용>
### 본문
<본문 심층 분석 내용>
"""


class State(TypedDict):
    """LangGraph conversation state storing accumulated messages."""

    messages: Annotated[list[AnyMessage], add_messages]


class ResearchRequest(BaseModel):
    """Request schema for initiating the journalist research pipeline."""

    about: str = Field(..., min_length=1, max_length=300, description="심층 분석 기사의 주제")


class ArticleOutput(BaseModel):
    """Structured news article output."""

    title: str
    subtitle: str
    lead: str
    body: str
    raw_markdown: str


class ResearchStep(BaseModel):
    """Single execution trace entry from LangGraph `.stream()`."""

    node: str
    summary: str
    tool_calls: list[str] = Field(default_factory=list)


class ResearchResponse(BaseModel):
    """Standardized API envelope returned to GAS or web clients."""

    ok: bool
    mode: Literal["live", "mock"]
    data: dict[str, Any] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Server health and configuration status response."""

    ok: bool
    google_api_configured: bool
    search_provider: str
    model: str


@tool
def get_current_time() -> str:
    """Return the current date and time in Asia/Seoul (KST) timezone."""
    now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
    return now_kst.strftime("%Y-%m-%d %H:%M:%S KST")


def _search_via_tavily(query: str, api_key: str) -> list[dict[str, str]]:
    """Query Tavily Search API and normalize results."""
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": 4, "include_answer": False},
        )
        resp.raise_for_status()
        payload = resp.json()
    return [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("url", "")),
            "snippet": str(item.get("content", ""))[:300],
        }
        for item in payload.get("results", [])
    ]


def _search_via_duckduckgo(query: str) -> list[dict[str, str]]:
    """Fallback instant-answer search via DuckDuckGo public JSON endpoint."""
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
        )
        resp.raise_for_status()
        payload = resp.json()

    results: list[dict[str, str]] = []
    if payload.get("AbstractText"):
        results.append(
            {
                "title": str(payload.get("Heading") or query),
                "url": str(payload.get("AbstractURL") or "https://duckduckgo.com"),
                "snippet": str(payload.get("AbstractText"))[:300],
            }
        )
    for topic in payload.get("RelatedTopics", [])[:3]:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append(
                {
                    "title": str(topic.get("Text"))[:60],
                    "url": str(topic.get("FirstURL") or "https://duckduckgo.com"),
                    "snippet": str(topic.get("Text"))[:300],
                }
            )
    return results


@tool
def get_web_search(query: str) -> str:
    """Search the web for recent news, policy documents, and supporting data for `query`."""
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    try:
        if tavily_key:
            results = _search_via_tavily(query, tavily_key)
        else:
            results = _search_via_duckduckgo(query)
        if not results:
            results = [
                {
                    "title": f"{query} 관련 쟁점 및 보도자료 요약",
                    "url": "https://news.google.com",
                    "snippet": f"'{query}'와 관련한 정책 당국·산업계·시민사회 간의 이해관계 충돌 및 제도적 보완 요구가 지속되고 있습니다.",
                }
            ]
        return json.dumps({"query": query, "results": results}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "query": query,
                "results": [
                    {
                        "title": f"{query} 검색 폴백 요약",
                        "url": "",
                        "snippet": f"실시간 검색 호출 오류({exc})로 인해 로컬 요약 컨텍스트를 반환합니다.",
                    }
                ],
            },
            ensure_ascii=False,
        )


class BasicToolNode:
    """Tool execution node that dispatches AIMessage `tool_calls` to registered tools."""

    def __init__(self, tools: list[Any]) -> None:
        self.tools_by_name: dict[str, Any] = {tool_item.name: tool_item for tool_item in tools}

    def __call__(self, inputs: dict[str, Any]) -> dict[str, list[ToolMessage]]:
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("No messages found in BasicToolNode input.")

        last_message = messages[-1]
        outputs: list[ToolMessage] = []
        for tool_call in getattr(last_message, "tool_calls", []) or []:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_fn = self.tools_by_name.get(tool_name)
            if tool_fn is None:
                result_str = json.dumps({"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False)
            else:
                result = tool_fn.invoke(tool_args)
                result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

            outputs.append(
                ToolMessage(
                    content=result_str,
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


TOOLS = [get_current_time, get_web_search]


def _get_model_name() -> str:
    """Resolve the active Gemini model name from `GEMINI_MODEL` or default."""
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def _build_llm_with_tools() -> Any:
    """Instantiate Google Gemini chat model bound with research tools."""
    llm = ChatGoogleGenerativeAI(model=_get_model_name(), temperature=0.3)
    return llm.bind_tools(TOOLS)


def generate(state: State) -> dict[str, list[AnyMessage]]:
    """LangGraph node that generates the journalist's next reasoning step, tool call, or article."""
    llm_with_tools = _build_llm_with_tools()
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def route_tools(state: State) -> Literal["tools", "__end__"]:
    """Route to `tools` when the last AI message contains tool calls, otherwise terminate."""
    messages = state.get("messages", [])
    if not messages:
        return END
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def build_journalist_graph() -> Any:
    """Compile the LangGraph workflow (`START -> generate -> route_tools -> tools / END`)."""
    graph_builder = StateGraph(State)
    graph_builder.add_node("generate", generate)
    graph_builder.add_node("tools", BasicToolNode(TOOLS))

    graph_builder.add_edge(START, "generate")
    graph_builder.add_conditional_edges(
        "generate",
        route_tools,
        {"tools": "tools", END: END},
    )
    graph_builder.add_edge("tools", "generate")
    return graph_builder.compile()


JOURNALIST_GRAPH = build_journalist_graph()


def _extract_text_content(content: Any) -> str:
    """Normalize LangChain message content (string or multimodal list) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ]
        return "\n".join(part for part in parts if part)
    return str(content or "")


def parse_article_markdown(raw_text: str, about: str) -> ArticleOutput:
    """Extract structured title, subtitle, lead, and body sections from model output."""
    title_match = re.search(r"^#\s*제목\s*[:：]\s*(.+)$", raw_text, re.MULTILINE)
    subtitle_match = re.search(r"^##\s*부제\s*[:：]\s*(.+)$", raw_text, re.MULTILINE)
    lead_match = re.search(r"###\s*리드문\s*\n+([\s\S]*?)(?=\n###\s*본문|\Z)", raw_text)
    body_match = re.search(r"###\s*본문\s*\n+([\s\S]*)\Z", raw_text)

    title = title_match.group(1).strip() if title_match else f"'{about}' 심층 비판 분석 리포트"
    subtitle = subtitle_match.group(1).strip() if subtitle_match else "쟁점별 구조적 한계와 향후 제도적 과제"
    lead = lead_match.group(1).strip() if lead_match else raw_text.strip().splitlines()[0] if raw_text.strip() else ""
    body = body_match.group(1).strip() if body_match else raw_text.strip()

    return ArticleOutput(
        title=title,
        subtitle=subtitle,
        lead=lead,
        body=body,
        raw_markdown=raw_text,
    )


def _build_initial_messages(about: str) -> list[AnyMessage]:
    """Create initial SystemMessage and HumanMessage for the journalist persona."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(about=about)
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"'{about}' 주제를 중심으로 단계별 검색과 목차 구성, 재검색을 거쳐 최종 심층 비판 기사를 작성해 주세요."
        ),
    ]


app = FastAPI(title="Veritas Editorial AI — LangGraph Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return server health and environment readiness."""
    return HealthResponse(
        ok=True,
        google_api_configured=bool(os.getenv("GOOGLE_API_KEY", "").strip()),
        search_provider="tavily" if os.getenv("TAVILY_API_KEY", "").strip() else "duckduckgo-fallback",
        model=_get_model_name(),
    )


@app.post("/api/research/stream")
def stream_research(req: ResearchRequest) -> StreamingResponse:
    """Stream LangGraph node updates (`generate`, `tools`) as NDJSON events using `.stream()`."""
    about = req.about.strip()
    if not os.getenv("GOOGLE_API_KEY", "").strip():
        def _error_gen():
            yield json.dumps(
                {"ok": False, "error": "GOOGLE_API_KEY 환경변수가 설정되지 않았습니다."},
                ensure_ascii=False,
            ) + "\n"

        return StreamingResponse(_error_gen(), media_type="application/x-ndjson", status_code=400)

    def _event_generator():
        initial_state: State = {"messages": _build_initial_messages(about)}
        for event in JOURNALIST_GRAPH.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                msgs = node_output.get("messages", [])
                last_msg = msgs[-1] if msgs else None
                tool_calls = [tc["name"] for tc in getattr(last_msg, "tool_calls", []) or []]
                summary = _extract_text_content(getattr(last_msg, "content", ""))
                yield json.dumps(
                    {"node": node_name, "tool_calls": tool_calls, "content": summary},
                    ensure_ascii=False,
                ) + "\n"

    return StreamingResponse(_event_generator(), media_type="application/x-ndjson")


@app.post("/api/research", response_model=ResearchResponse)
def run_research(req: ResearchRequest) -> ResearchResponse:
    """Consume LangGraph `.stream()` updates and return steps + structured article for GAS clients."""
    about = req.about.strip()
    if not os.getenv("GOOGLE_API_KEY", "").strip():
        return ResearchResponse(
            ok=False,
            mode="live",
            data=None,
            error="서버에 GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.",
        )

    try:
        initial_state: State = {"messages": _build_initial_messages(about)}
        steps: list[ResearchStep] = []
        final_text = ""

        for event in JOURNALIST_GRAPH.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                msgs = node_output.get("messages", [])
                for msg in msgs:
                    content_text = _extract_text_content(getattr(msg, "content", ""))
                    tool_calls = [tc["name"] for tc in getattr(msg, "tool_calls", []) or []]
                    summary_preview = content_text[:280] if content_text else ("도구 호출: " + ", ".join(tool_calls))
                    steps.append(ResearchStep(node=node_name, summary=summary_preview, tool_calls=tool_calls))
                    if node_name == "generate" and not tool_calls and content_text:
                        final_text = content_text

        article = parse_article_markdown(final_text, about)
        return ResearchResponse(
            ok=True,
            mode="live",
            data={
                "topic": about,
                "steps": [step.model_dump() for step in steps],
                "article": article.model_dump(),
            },
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return ResearchResponse(ok=False, mode="live", data=None, error=f"LangGraph 실행 실패: {exc}")
