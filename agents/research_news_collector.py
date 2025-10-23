"""
Module: agents/research_news_collector.py
Purpose: 최신 연구/뉴스 수집 에이전트 (LangGraph 기반)

기능:
- 최근 1~2년간 주요 연구 결과 및 뉴스 기사를 출처별로 수집
- Tavily 검색 API를 통한 웹 검색
- 시간 범위 및 키워드 기반 필터링
- Pydantic 스키마 기반 구조화된 출력

입력 State:
state = {
    "topic": "AI 기술",              # (필수) 검색 주제/키워드
    "time_range": "1-2 years",      # (선택) 시간 범위
    "max_results": 20,              # (선택) 최대 결과 수
    "search_depth": "advanced"      # (선택) 검색 깊이 (basic|advanced)
}

출력 State:
state["research_news"] = {
    "query": "...",
    "articles": [
        {
            "title": "...",
            "url": "...",
            "source": "...",
            "published_date": "...",
            "summary": "...",
            "relevance_score": 0.95,
            "category": "research|news"
        },
        ...
    ],
    "sources": [...],
    "collection_date": "2025-01-20"
}

환경 변수:
- TAVILY_API_KEY (필수)
- OPENAI_API_KEY (선택 - 요약 및 분류)
"""
from __future__ import annotations

import os
import json
import time
from datetime import datetime, timedelta
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
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# Pydantic
try:
    from pydantic import BaseModel, Field, ConfigDict
    PYDANTIC_AVAILABLE = True
except Exception:
    PYDANTIC_AVAILABLE = False

# Tavily
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except Exception as e:
    TAVILY_AVAILABLE = False
    print(f"⚠️ Tavily import 실패: {e}")


# ----------------------------- Pydantic Models -----------------------------

if PYDANTIC_AVAILABLE:
    class Article(BaseModel):
        """연구/뉴스 기사"""
        title: str
        url: str
        source: str
        published_date: Optional[str] = ""
        summary: str
        relevance_score: Optional[float] = 0.0
        category: str = "news"  # research | news
        
        model_config = ConfigDict(str_strip_whitespace=True)
    
    class ArticleList(BaseModel):
        """기사 목록"""
        items: List[Article] = []
else:
    Article = dict
    ArticleList = dict


# ----------------------------- State & Config -----------------------------

class ResearchNewsOutput(TypedDict, total=False):
    query: str
    articles: List[Dict[str, Any]]
    sources: List[str]
    collection_date: str

class ResearchNewsState(TypedDict, total=False):
    topic: str
    time_range: str
    max_results: int
    search_depth: str
    research_news: ResearchNewsOutput

@dataclass
class ResearchNewsConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_results: int = 20
    search_depth: str = "advanced"
    use_categorization: bool = True  # LLM으로 연구/뉴스 분류


# ----------------------------- Utilities -----------------------------

def _norm(text: str) -> str:
    return (text or "").strip()

def _extract_date_from_text(text: str) -> Optional[str]:
    """텍스트에서 날짜 추출 (간단한 휴리스틱)"""
    # 2024, 2025 등의 연도 찾기
    import re
    year_match = re.search(r'20(2[0-5]|1[0-9])', text)
    if year_match:
        return year_match.group(0)
    return None

def _calculate_relevance_score(title: str, content: str, keywords: List[str]) -> float:
    """키워드 기반 관련성 점수 계산"""
    text = f"{title} {content}".lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in text)
    max_score = len(keywords)
    if max_score == 0:
        return 0.5
    return min(1.0, hits / max_score)

def _parse_time_range_to_years(time_range: str) -> str:
    """
    시간 범위를 연도 범위로 변환
    
    예시:
    - "1-2 years" → "2024 2025"
    - "3-4 years" → "2021 2022 2023 2024 2025"
    - "4-5 years" → "2020 2021 2022 2023 2024 2025"
    """
    import re
    from datetime import datetime
    
    current_year = datetime.now().year
    
    # "X-Y years" 형식에서 숫자 추출
    match = re.search(r'(\d+)-?(\d+)?\s*years?', time_range.lower())
    
    if match:
        start_years = int(match.group(1))
        end_years = int(match.group(2)) if match.group(2) else start_years
        max_years = max(start_years, end_years)
        
        # 연도 범위 생성 (과거부터 현재까지)
        start_year = current_year - max_years
        years = list(range(start_year, current_year + 1))
        
        return " ".join(map(str, years))
    
    # 기본값: 최근 2년
    return f"{current_year - 1} {current_year}"


# ----------------------------- Prompts (외부 파일에서 로드) -----------------------------

try:
    from prompts.research_news_prompts import CATEGORIZATION_PROMPT
except ImportError:
    # 폴백: 프롬프트를 찾을 수 없는 경우 기본값 사용
    CATEGORIZATION_PROMPT = """당신은 기술 뉴스 및 연구 자료 분류 전문가입니다.
기사를 'research' 또는 'news'로 분류하세요.
JSON 형식으로 반환: {{"category": "research" 또는 "news", "confidence": 0.0~1.0}}
"""


# ----------------------------- Node Factory -----------------------------

def _research_news_node_factory(cfg: ResearchNewsConfig):
    """LangGraph 노드 함수 팩토리"""
    
    def node(state: ResearchNewsState) -> ResearchNewsState:
        topic = _norm(state.get("topic", "AI 기술"))
        time_range = _norm(state.get("time_range", "1-2 years"))
        max_results = int(state.get("max_results", cfg.max_results))
        search_depth = _norm(state.get("search_depth", cfg.search_depth))
        
        print(f"\n🔍 연구/뉴스 수집 시작: {topic}")
        print(f"   - 시간 범위: {time_range}")
        print(f"   - 최대 결과: {max_results}")
        print(f"   - 검색 깊이: {search_depth}")
        
        # time_range를 연도 범위로 변환
        year_range_str = _parse_time_range_to_years(time_range)
        print(f"   - 검색 연도: {year_range_str}")
        
        # 검색 쿼리 구성 (연구/뉴스 균형)
        research_queries = [
            f"{topic} 최신 연구 논문 {year_range_str}",
            f"{topic} research paper {year_range_str}",
        ]
        
        news_queries = [
            f"{topic} 뉴스 기사 {year_range_str}",
            f"{topic} news article {year_range_str}",
            f"{topic} 산업 동향 최신 뉴스",
            f"{topic} 시장 전망 업계 뉴스",
        ]
        
        articles: List[Dict[str, Any]] = []
        sources_set = set()
        
        tavily_key = os.getenv("TAVILY_API_KEY")
        
        if TAVILY_AVAILABLE and tavily_key:
            try:
                print(f"   🌐 Tavily 웹 검색 모드 활성화")
                client = TavilyClient(api_key=tavily_key)
                
                # 1. 연구 논문 검색
                print(f"   📄 [1/2] 연구 논문 검색...")
                for query in research_queries[:2]:
                    try:
                        print(f"      📡 {query}")
                        response = client.search(
                            query=query,
                            max_results=min(max_results // 2, 10),
                            search_depth=search_depth,
                            include_domains=["arxiv.org", "papers.ssrn.com", "ieee.org", "acm.org", "scholar.google.com"]
                        )
                        
                        results = response.get("results", [])
                        print(f"         ✅ {len(results)}개 발견")
                        
                        for r in results:
                            title = _norm(r.get("title", ""))
                            url = _norm(r.get("url", ""))
                            content = _norm(r.get("content", ""))
                            
                            if not title or not url:
                                continue
                            
                            from urllib.parse import urlparse
                            domain = urlparse(url).netloc
                            sources_set.add(domain)
                            
                            published_date = _extract_date_from_text(content) or datetime.now().strftime("%Y")
                            keywords = topic.split() + ["AI", "기술", "연구", "technology"]
                            relevance = _calculate_relevance_score(title, content, keywords)
                            
                            article = {
                                "title": title,
                                "url": url,
                                "source": domain,
                                "published_date": published_date,
                                "summary": content[:300],
                                "relevance_score": round(relevance, 3),
                                "category": "research"
                            }
                            articles.append(article)
                        
                    except Exception as e:
                        print(f"         ⚠️ 검색 실패: {e}")
                        continue
                
                # 2. 뉴스 기사 검색 (도메인 제한 없이 폭넓게 검색)
                print(f"   📰 [2/2] 뉴스 기사 검색...")
                for query in news_queries[:2]:
                    try:
                        print(f"      📡 {query}")
                        response = client.search(
                            query=query,
                            max_results=min(max_results // 2, 10),
                            search_depth=search_depth
                            # include_domains 제거 - 뉴스는 다양한 출처에서 수집
                        )
                        
                        results = response.get("results", [])
                        print(f"         ✅ {len(results)}개 발견")
                        
                        for r in results:
                            title = _norm(r.get("title", ""))
                            url = _norm(r.get("url", ""))
                            content = _norm(r.get("content", ""))
                            
                            if not title or not url:
                                continue
                            
                            # 출처 추출
                            from urllib.parse import urlparse
                            domain = urlparse(url).netloc
                            sources_set.add(domain)
                            
                            # 날짜 추출 (간단히 현재 날짜 사용)
                            published_date = _extract_date_from_text(content) or datetime.now().strftime("%Y")
                            
                            # 관련성 점수
                            keywords = topic.split() + ["뉴스", "산업", "시장", "news", "industry", "market"]
                            relevance = _calculate_relevance_score(title, content, keywords)
                            
                            # 뉴스 검색이므로 카테고리는 무조건 "news"
                            article = {
                                "title": title,
                                "url": url,
                                "source": domain,
                                "published_date": published_date,
                                "summary": content[:300],
                                "relevance_score": round(relevance, 3),
                                "category": "news"
                            }
                            
                            articles.append(article)
                        
                    except Exception as e:
                        print(f"         ⚠️ 검색 실패: {e}")
                        continue
                
            except Exception as e:
                print(f"⚠️ Tavily 클라이언트 초기화 실패: {e}")
        else:
            print(f"   ⚠️ Tavily API 사용 불가 - 레거시 더미 데이터 모드")
            if not tavily_key:
                print(f"   원인: TAVILY_API_KEY 미설정")
            if not TAVILY_AVAILABLE:
                print(f"   원인: tavily-python 라이브러리 미설치")
            # 더미 데이터
            articles = [
                {
                    "title": f"{topic} 관련 최신 연구 동향",
                    "url": "https://example.com/research1",
                    "source": "example.com",
                    "published_date": "2024",
                    "summary": f"{topic}에 대한 최신 연구 동향을 다룬 기사입니다.",
                    "relevance_score": 0.9,
                    "category": "research"
                },
                {
                    "title": f"{topic} 산업 뉴스",
                    "url": "https://example.com/news1",
                    "source": "example.com",
                    "published_date": "2025",
                    "summary": f"{topic} 산업의 최신 동향을 전합니다.",
                    "relevance_score": 0.85,
                    "category": "news"
                }
            ]
            sources_set = {"example.com"}
        
        # 중복 제거 (URL 기준)
        seen_urls = set()
        unique_articles = []
        for article in articles:
            if article["url"] not in seen_urls:
                seen_urls.add(article["url"])
                unique_articles.append(article)
        
        # 관련성 점수로 정렬
        unique_articles.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        unique_articles = unique_articles[:max_results]
        
        state["research_news"] = {
            "query": topic,
            "articles": unique_articles,
            "sources": sorted(list(sources_set)),
            "collection_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        # 결과 출력 (간소화)
        research_count = sum(1 for a in unique_articles if a.get("category") == "research")
        news_count = len(unique_articles) - research_count
        
        print(f"\n   ✅ 수집 완료: 총 {len(unique_articles)}개 (연구: {research_count}, 뉴스: {news_count})")
        print(f"   📚 출처: {len(sources_set)}개")
        
        # 상위 2개만 미리보기
        for idx, article in enumerate(unique_articles[:2], 1):
            print(f"   [{idx}] {article.get('title', 'N/A')[:60]}...")
        
        if len(unique_articles) > 2:
            print(f"   ... 외 {len(unique_articles)-2}개")
        
        return state
    
    return node


# ----------------------------- Graph Builder -----------------------------

def build_research_news_graph(config: Optional[ResearchNewsConfig] = None):
    cfg = config or ResearchNewsConfig()
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph가 설치되어 있지 않습니다. `pip install langgraph` 후 다시 시도하세요.")
    
    g = StateGraph(ResearchNewsState)
    g.add_node("research_news_collector", _research_news_node_factory(cfg))
    g.add_edge(START, "research_news_collector")
    g.add_edge("research_news_collector", END)
    return g.compile()


# ----------------------------- Helper -----------------------------

def run_research_news_collector(state: Dict[str, Any], config: Optional[ResearchNewsConfig] = None) -> Dict[str, Any]:
    """연구/뉴스 수집 실행"""
    app = build_research_news_graph(config)
    return app.invoke(state)


# ----------------------------- Output Formatting -----------------------------

def print_research_news_results(result: Dict[str, Any]):
    """수집 결과를 보기 좋게 출력"""
    research_news = result.get("research_news", {})
    articles = research_news.get("articles", [])
    
    print("\n" + "=" * 80)
    print("📰 연구/뉴스 수집 결과")
    print("=" * 80)
    print(f"\n검색 주제: {research_news.get('query', 'N/A')}")
    print(f"수집 날짜: {research_news.get('collection_date', 'N/A')}")
    print(f"총 수집: {len(articles)}개")
    print(f"출처: {', '.join(research_news.get('sources', []))}")
    print("\n" + "-" * 80)
    
    for idx, article in enumerate(articles, 1):
        print(f"\n[{idx}] {article.get('title', 'Unknown')}")
        print(f"    📂 유형: {article.get('category', 'N/A')}")
        print(f"    📅 날짜: {article.get('published_date', 'N/A')}")
        print(f"    🔗 URL: {article.get('url', 'N/A')}")
        print(f"    ⭐ 관련성: {article.get('relevance_score', 0):.2f}")
        print(f"    📝 요약: {article.get('summary', '')[:150]}...")
    
    print("\n" + "=" * 80)


# ----------------------------- CLI Test -----------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="연구/뉴스 수집 에이전트 (LangGraph)")
    parser.add_argument("--topic", default="AI 기술 트렌드", help="검색 주제")
    parser.add_argument("--time-range", default="1-2 years", help="시간 범위")
    parser.add_argument("--max-results", type=int, default=15, help="최대 결과 수")
    parser.add_argument("--depth", default="advanced", help="검색 깊이 (basic|advanced)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("📰 연구/뉴스 수집 에이전트")
    print("=" * 80)
    print(f"\n설정:")
    print(f"  - 주제: {args.topic}")
    print(f"  - 시간 범위: {args.time_range}")
    print(f"  - 최대 결과: {args.max_results}")
    print(f"  - 검색 깊이: {args.depth}")
    
    initial: ResearchNewsState = {
        "topic": args.topic,
        "time_range": args.time_range,
        "max_results": args.max_results,
        "search_depth": args.depth
    }
    
    config = ResearchNewsConfig(
        max_results=args.max_results,
        search_depth=args.depth
    )
    
    try:
        final = run_research_news_collector(initial, config)
        print_research_news_results(final)
        
        # JSON 저장
        output_file = "research_news_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final.get("research_news"), f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과가 저장되었습니다: {output_file}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

