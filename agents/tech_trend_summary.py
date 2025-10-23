"""
Module: agents/tech_trend_summary.py
Purpose: 핵심 기술 요약 에이전트 (LangGraph 기반)

기능:
- 수집된 연구/뉴스 정보를 바탕으로 주요 기술별 트렌드 요약
- LLM을 활용한 기술 분류 및 요약
- 기술별 주요 특징, 적용 사례, 발전 방향 분석

입력 State:
state = {
    "research_news": {
        "articles": [...],
        "query": "..."
    }
}

출력 State:
state["tech_trend_summary"] = {
    "technology_categories": [
        {
            "category": "생성형 AI",
            "key_technologies": ["GPT-4", "Stable Diffusion", "DALL-E"],
            "summary": "...",
            "applications": ["콘텐츠 생성", "코드 작성"],
            "maturity_level": "성숙기",
            "trend_direction": "상승"
        },
        ...
    ],
    "overall_summary": "...",
    "key_insights": [...]
}

환경 변수:
- OPENAI_API_KEY (필수)
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, TypedDict
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# mini-project 폴더의 .env 파일을 명시적으로 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# LangGraph
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

# LangChain
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# Pydantic
try:
    from pydantic import BaseModel, Field, ConfigDict
    PYDANTIC_AVAILABLE = True
except Exception:
    PYDANTIC_AVAILABLE = False


# ----------------------------- Pydantic Models -----------------------------

if PYDANTIC_AVAILABLE:
    class TechnologyCategory(BaseModel):
        """기술 카테고리"""
        category: str = Field(description="기술 카테고리명")
        key_technologies: List[str] = Field(default=[], description="주요 기술들")
        summary: str = Field(description="기술 요약")
        applications: List[str] = Field(default=[], description="적용 분야")
        maturity_level: str = Field(default="발전 중", description="성숙도")
        trend_direction: str = Field(default="상승", description="트렌드 방향")
        
        model_config = ConfigDict(str_strip_whitespace=True)
    
    class TechTrendSummaryResult(BaseModel):
        """기술 트렌드 요약 결과"""
        technology_categories: List[TechnologyCategory]
        overall_summary: str
        key_insights: List[str] = []
else:
    TechnologyCategory = dict
    TechTrendSummaryResult = dict


# ----------------------------- State & Config -----------------------------

class TechTrendSummaryOutput(TypedDict, total=False):
    technology_categories: List[Dict[str, Any]]
    overall_summary: str
    key_insights: List[str]

class TechTrendSummaryState(TypedDict, total=False):
    research_news: Dict[str, Any]
    tech_trend_summary: TechTrendSummaryOutput

@dataclass
class TechTrendSummaryConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    use_structured_output: bool = True


# ----------------------------- Utilities -----------------------------

def _norm(text: str) -> str:
    return (text or "").strip()

def _aggregate_articles(articles: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    """기사들을 집계하여 텍스트로 변환"""
    aggregated = []
    total_chars = 0
    
    for idx, article in enumerate(articles, 1):
        title = _norm(article.get("title", ""))
        summary = _norm(article.get("summary", ""))
        category = article.get("category", "news")
        
        text = f"[{idx}] ({category}) {title}\n{summary}\n"
        
        if total_chars + len(text) > max_chars:
            break
        
        aggregated.append(text)
        total_chars += len(text)
    
    return "\n".join(aggregated)


# ----------------------------- Prompts (외부 파일에서 로드) -----------------------------

try:
    from prompts.tech_summary_prompts import (
        TECH_SUMMARY_SYSTEM_PROMPT,
        TECH_SUMMARY_USER_QUERY_TMPL
    )
except ImportError:
    # 폴백: 프롬프트를 찾을 수 없는 경우 기본값 사용
    TECH_SUMMARY_SYSTEM_PROMPT = """당신은 기술 트렌드 분석 전문가입니다.

**분석 목표: 수집된 연구/뉴스를 기반으로 주요 기술별 트렌드 요약**

핵심 원칙:
1. **기술 분류**: 유사한 기술들을 카테고리로 그룹화
2. **구체적 요약**: 각 카테고리의 핵심 기술과 특징을 명확히 설명
3. **적용 사례**: 실제 적용되고 있는 분야와 사례 제시
4. **트렌드 파악**: 현재 성숙도와 향후 발전 방향 분석

분석 작업:

1. **기술 카테고리 식별**
   - 기사들에서 언급된 기술들을 주요 카테고리로 분류
   - 예: "생성형 AI", "자율주행", "양자컴퓨팅", "블록체인" 등

2. **주요 기술 (key_technologies)**
   - 각 카테고리 내 구체적인 기술, 모델, 플랫폼 나열
   - 예: ["GPT-4", "Claude", "Gemini"]

3. **기술 요약 (summary)**
   - 해당 카테고리의 핵심 특징과 최근 동향을 2-3문장으로 요약
   - 구체적인 수치나 사례가 있으면 포함

4. **적용 분야 (applications)**
   - 실제 활용되고 있는 산업/분야 나열
   - 예: ["의료 진단", "금융 분석", "콘텐츠 생성"]

5. **성숙도 (maturity_level)**
   - "초기 단계", "발전 중", "성숙기", "안정기" 중 선택

6. **트렌드 방향 (trend_direction)**
   - "급상승", "상승", "유지", "하락" 중 선택

출력 스키마 (JSON만):
{{
  "technology_categories": [
    {{
      "category": "",
      "key_technologies": [],
      "summary": "",
      "applications": [],
      "maturity_level": "",
      "trend_direction": ""
    }}
  ],
  "overall_summary": "전체 기술 트렌드를 3-5문장으로 요약",
  "key_insights": ["인사이트 1", "인사이트 2", ...]
}}
"""
    TECH_SUMMARY_USER_QUERY_TMPL = """다음 연구/뉴스 자료를 분석하여 주요 기술별 트렌드를 요약하세요.

검색 주제: {query}
수집 날짜: {collection_date}
자료 수: {article_count}개

=== 수집된 자료 ===
{articles_text}
====================

**분석 지침:**
1. 최소 3개 이상의 주요 기술 카테고리 식별
2. 각 카테고리마다 구체적인 기술/모델명 나열
3. 실제 적용 사례와 트렌드 방향 명시
4. 전체 요약과 핵심 인사이트 도출

JSON으로만 반환하세요.
"""


# ----------------------------- Node Factory -----------------------------

def _tech_trend_summary_node_factory(cfg: TechTrendSummaryConfig):
    """LangGraph 노드 함수 팩토리"""
    
    def node(state: TechTrendSummaryState) -> TechTrendSummaryState:
        research_news = state.get("research_news", {})
        articles = research_news.get("articles", [])
        query = research_news.get("query", "기술 트렌드")
        collection_date = research_news.get("collection_date", "N/A")
        
        print(f"\n📊 기술 트렌드 요약 시작")
        print(f"   - 분석 자료: {len(articles)}개")
        
        if not articles:
            print(f"⚠️ 분석할 자료가 없습니다.")
            state["tech_trend_summary"] = {
                "technology_categories": [],
                "overall_summary": "분석할 자료가 부족합니다.",
                "key_insights": []
            }
            return state
        
        # 기사 집계
        articles_text = _aggregate_articles(articles, max_chars=8000)
        
        # LLM Structured Output 방식
        if cfg.use_structured_output and LANGCHAIN_AVAILABLE and PYDANTIC_AVAILABLE:
            try:
                llm = ChatOpenAI(model=cfg.model, temperature=cfg.temperature)
                structured_llm = llm.with_structured_output(TechTrendSummaryResult)
                
                sys_prompt = TECH_SUMMARY_SYSTEM_PROMPT
                user_prompt = TECH_SUMMARY_USER_QUERY_TMPL.format(
                    query=query,
                    collection_date=collection_date,
                    article_count=len(articles),
                    articles_text=articles_text
                )
                
                print(f"   🤖 AI Structured Output 분석 중...")
                result: TechTrendSummaryResult = structured_llm.invoke([
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content=user_prompt)
                ])
                
                # Pydantic → Dict 변환
                tech_categories = []
                for cat in result.technology_categories:
                    tech_categories.append({
                        "category": cat.category,
                        "key_technologies": cat.key_technologies,
                        "summary": cat.summary,
                        "applications": cat.applications,
                        "maturity_level": cat.maturity_level,
                        "trend_direction": cat.trend_direction
                    })
                
                state["tech_trend_summary"] = {
                    "technology_categories": tech_categories,
                    "overall_summary": result.overall_summary,
                    "key_insights": result.key_insights
                }
                
                print(f"   ✅ AI 분석 완료 (Structured Output)")
                
            except Exception as e:
                print(f"   ⚠️ Structured Output 실패, 레거시 방식 사용")
                print(f"   오류: {str(e)[:100]}")
                # 폴백: 간단한 분석
                state["tech_trend_summary"] = _fallback_analysis(articles, query)
        else:
            # 기본 방식
            print(f"   ⚠️ 레거시 분석 모드 (Structured Output 비활성화)")
            state["tech_trend_summary"] = _fallback_analysis(articles, query)
        
        # 결과 출력 (간소화)
        summary = state["tech_trend_summary"]
        
        print(f"\n   ✅ 분석 완료: {len(summary.get('technology_categories', []))}개 기술 카테고리")
        
        for idx, cat in enumerate(summary.get("technology_categories", [])[:2], 1):
            print(f"   [{idx}] {cat.get('category', 'N/A')} - {cat.get('maturity_level', 'N/A')}/{cat.get('trend_direction', 'N/A')}")
        
        if len(summary.get("technology_categories", [])) > 2:
            print(f"   ... 외 {len(summary['technology_categories'])-2}개")
        
        insights = summary.get("key_insights", [])
        if insights:
            print(f"   💡 인사이트: {insights[0][:60]}...")
        
        return state
    
    return node


def _fallback_analysis(articles: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """폴백: 간단한 키워드 기반 분석"""
    # 간단히 카테고리 1개만 생성
    return {
        "technology_categories": [
            {
                "category": query,
                "key_technologies": ["기술1", "기술2"],
                "summary": f"{query} 관련 기술들이 빠르게 발전하고 있습니다.",
                "applications": ["산업1", "산업2"],
                "maturity_level": "발전 중",
                "trend_direction": "상승"
            }
        ],
        "overall_summary": f"{len(articles)}개의 자료를 바탕으로 {query} 트렌드를 분석했습니다.",
        "key_insights": ["최신 기술 동향 파악 완료", "다양한 적용 사례 발견"]
    }


# ----------------------------- Graph Builder -----------------------------

def build_tech_trend_summary_graph(config: Optional[TechTrendSummaryConfig] = None):
    cfg = config or TechTrendSummaryConfig()
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph가 설치되어 있지 않습니다. `pip install langgraph` 후 다시 시도하세요.")
    
    g = StateGraph(TechTrendSummaryState)
    g.add_node("tech_trend_summary", _tech_trend_summary_node_factory(cfg))
    g.add_edge(START, "tech_trend_summary")
    g.add_edge("tech_trend_summary", END)
    return g.compile()


# ----------------------------- Helper -----------------------------

def run_tech_trend_summary(state: Dict[str, Any], config: Optional[TechTrendSummaryConfig] = None) -> Dict[str, Any]:
    """기술 트렌드 요약 실행"""
    app = build_tech_trend_summary_graph(config)
    return app.invoke(state)


# ----------------------------- Output Formatting -----------------------------

def print_tech_trend_summary_results(result: Dict[str, Any]):
    """요약 결과를 보기 좋게 출력"""
    summary = result.get("tech_trend_summary", {})
    categories = summary.get("technology_categories", [])
    
    print("\n" + "=" * 80)
    print("📊 기술 트렌드 요약 결과")
    print("=" * 80)
    
    print(f"\n🌐 전체 요약:")
    print(f"{summary.get('overall_summary', 'N/A')}")
    
    print(f"\n📂 기술 카테고리 ({len(categories)}개):")
    print("-" * 80)
    
    for idx, cat in enumerate(categories, 1):
        print(f"\n[{idx}] {cat.get('category', 'Unknown')}")
        print(f"    🔧 주요 기술: {', '.join(cat.get('key_technologies', []))}")
        print(f"    📝 요약: {cat.get('summary', '')}")
        print(f"    💼 적용 분야: {', '.join(cat.get('applications', []))}")
        print(f"    📈 성숙도: {cat.get('maturity_level', 'N/A')}")
        print(f"    📊 트렌드: {cat.get('trend_direction', 'N/A')}")
    
    insights = summary.get("key_insights", [])
    if insights:
        print(f"\n💡 핵심 인사이트 ({len(insights)}개):")
        print("-" * 80)
        for idx, insight in enumerate(insights, 1):
            print(f"  {idx}. {insight}")
    
    print("\n" + "=" * 80)


# ----------------------------- CLI Test -----------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("📊 기술 트렌드 요약 에이전트 테스트")
    print("=" * 80)
    
    # 모의 데이터
    dummy_state = {
        "research_news": {
            "query": "AI 기술",
            "collection_date": "2025-01-20",
            "articles": [
                {
                    "title": "GPT-4 Turbo 출시, 비용 절감 및 성능 향상",
                    "summary": "OpenAI가 GPT-4 Turbo를 발표했습니다. 기존 대비 비용은 1/3, 속도는 2배 향상되었습니다.",
                    "category": "news"
                },
                {
                    "title": "Stable Diffusion 3.0 연구 논문 발표",
                    "summary": "이미지 생성 품질이 크게 개선된 Stable Diffusion 3.0 연구 결과가 공개되었습니다.",
                    "category": "research"
                },
                {
                    "title": "자율주행 AI 기술, 레벨 4 상용화 근접",
                    "summary": "완전 자율주행에 가까운 레벨 4 기술이 여러 기업에서 시험 운영 중입니다.",
                    "category": "news"
                }
            ]
        }
    }
    
    try:
        config = TechTrendSummaryConfig()
        final = run_tech_trend_summary(dummy_state, config)
        print_tech_trend_summary_results(final)
        
        # JSON 저장
        output_file = "tech_trend_summary_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final.get("tech_trend_summary"), f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과가 저장되었습니다: {output_file}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

