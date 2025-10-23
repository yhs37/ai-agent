"""
Module: main_graph.py
Purpose: 기술 트렌드 분석 통합 그래프 (LangGraph 기반)

5개 에이전트를 순차적으로 실행하는 통합 워크플로우:
1. 연구/뉴스 수집 에이전트
2. 핵심 기술 요약 에이전트
3. 트렌드 예측 에이전트
4. 리스크 및 기회 분석 에이전트
5. 트렌드 보고서 작성 에이전트

실행 방법:
```bash
python main_graph.py --topic "AI 기술 트렌드" --max-results 15
```

환경 변수:
- TAVILY_API_KEY (필수 - 뉴스 수집)
- OPENAI_API_KEY (필수 - 분석 및 예측)
"""
from __future__ import annotations

import os
import argparse
from typing import Any, Dict, TypedDict
from pathlib import Path
from dotenv import load_dotenv

# mini-project 폴더의 .env 파일을 명시적으로 로드
current_dir = Path(__file__).parent
env_path = current_dir / ".env"
load_dotenv(dotenv_path=env_path, override=True)

print(f"🔧 .env 파일 로드: {env_path}")
print(f"   OPENAI_API_KEY: {'설정됨' if os.getenv('OPENAI_API_KEY') else '없음'}")
print(f"   TAVILY_API_KEY: {'설정됨' if os.getenv('TAVILY_API_KEY') else '없음'}")

# LangGraph
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    print(f"⚠️ LangGraph import 실패: {e}")
    print("설치: pip install langgraph")

# 각 에이전트 import
try:
    from agents.research_news_collector import (
        _research_news_node_factory,
        ResearchNewsConfig,
        print_research_news_results
    )
    from agents.tech_trend_summary import (
        _tech_trend_summary_node_factory,
        TechTrendSummaryConfig,
        print_tech_trend_summary_results
    )
    from agents.trend_prediction import (
        _trend_prediction_node_factory,
        TrendPredictionConfig,
        print_trend_prediction_results
    )
    from agents.risk_opportunity_analysis import (
        _risk_opportunity_node_factory,
        RiskOpportunityConfig,
        print_risk_opportunity_results
    )
    from agents.trend_report_generator import (
        _trend_report_node_factory,
        TrendReportConfig,
        save_report_markdown,
        print_trend_report_summary
    )
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENTS_AVAILABLE = False
    print(f"⚠️ 에이전트 import 실패: {e}")
    print("agents 폴더의 모든 에이전트 파일이 필요합니다.")


# ----------------------------- 통합 State -----------------------------

class TrendAnalysisState(TypedDict, total=False):
    """전체 워크플로우의 통합 State"""
    # 입력 파라미터
    topic: str
    time_range: str
    max_results: int
    search_depth: str
    
    # 각 에이전트의 출력
    research_news: Dict[str, Any]
    tech_trend_summary: Dict[str, Any]
    trend_prediction: Dict[str, Any]
    risk_opportunity: Dict[str, Any]
    trend_report: Dict[str, Any]


# ----------------------------- State 출력 Wrapper -----------------------------

def create_logging_node(node_func, node_name: str, agent_name: str):
    """노드 실행 전후에 state를 출력하는 래퍼"""
    def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n" + "🔹"*40)
        print(f"📍 [{agent_name}] 시작")
        print("🔹"*40)
        
        # 입력 state 출력
        print(f"\n📥 입력 State 키: {list(state.keys())}")
        if 'topic' in state:
            print(f"   - 분석 주제: {state.get('topic', 'N/A')}")
        if 'research_news' in state:
            articles_count = len(state.get('research_news', {}).get('articles', []))
            print(f"   - 수집된 자료: {articles_count}개")
        if 'tech_trend_summary' in state:
            categories_count = len(state.get('tech_trend_summary', {}).get('technology_categories', []))
            print(f"   - 기술 카테고리: {categories_count}개")
        if 'trend_prediction' in state:
            predictions_count = len(state.get('trend_prediction', {}).get('predictions', []))
            print(f"   - 예측 대상: {predictions_count}개")
        if 'risk_opportunity' in state:
            analyses_count = len(state.get('risk_opportunity', {}).get('analyses', []))
            print(f"   - 분석 대상: {analyses_count}개")
        
        print()
        
        # 노드 실행
        result = node_func(state)
        
        # 출력 state 출력
        print(f"\n📤 출력 State 키: {list(result.keys())}")
        print(f"🔹"*40)
        print(f"✅ [{agent_name}] 완료\n")
        
        return result
    
    return wrapper


# ----------------------------- 통합 그래프 빌더 -----------------------------

# ----------------------------- 순차 실행 방식 (LangGraph 호환성 문제 해결) -----------------------------

def run_sequential_workflow(
    initial_state: Dict[str, Any],
    config_research: ResearchNewsConfig,
    config_tech_summary: TechTrendSummaryConfig,
    config_prediction: TrendPredictionConfig,
    config_risk_opp: RiskOpportunityConfig,
    config_report: TrendReportConfig
) -> Dict[str, Any]:
    """
    LangGraph 그래프 대신 순차 실행 방식 사용
    (버전 호환성 문제 해결)
    """
    
    state = initial_state.copy()
    
    # 1. 연구/뉴스 수집
    node_func_1 = create_logging_node(
        _research_news_node_factory(config_research),
        "collect_research_news",
        "1/5 연구/뉴스 수집 에이전트"
    )
    state = node_func_1(state)
    
    # 2. 기술 트렌드 요약
    node_func_2 = create_logging_node(
        _tech_trend_summary_node_factory(config_tech_summary),
        "summarize_tech_trends",
        "2/5 기술 트렌드 요약 에이전트"
    )
    state = node_func_2(state)
    
    # 3. 트렌드 예측
    node_func_3 = create_logging_node(
        _trend_prediction_node_factory(config_prediction),
        "predict_trends",
        "3/5 트렌드 예측 에이전트"
    )
    state = node_func_3(state)
    
    # 4. 리스크/기회 분석
    node_func_4 = create_logging_node(
        _risk_opportunity_node_factory(config_risk_opp),
        "analyze_risk_opportunity",
        "4/5 리스크/기회 분석 에이전트"
    )
    state = node_func_4(state)
    
    # 5. 보고서 생성
    node_func_5 = create_logging_node(
        _trend_report_node_factory(config_report),
        "generate_report",
        "5/5 보고서 생성 에이전트"
    )
    state = node_func_5(state)
    
    return state


# ----------------------------- 실행 함수 -----------------------------

def run_trend_analysis(
    topic: str = "AI 기술 트렌드",
    time_range: str = "1-2 years",
    max_results: int = 15,
    search_depth: str = "advanced",
    output_dir: str = "output"
) -> Dict[str, Any]:
    """
    기술 트렌드 분석 전체 워크플로우 실행
    
    Args:
        topic: 분석 주제
        time_range: 시간 범위
        max_results: 최대 수집 결과 수
        search_depth: 검색 깊이 (basic|advanced)
        output_dir: 출력 디렉토리
    
    Returns:
        전체 분석 결과 State
    """
    
    print("\n" + "="*80)
    print("🌐 기술 트렌드 분석 워크플로우 시작")
    print("="*80)
    print(f"\n📋 분석 설정:")
    print(f"   - 주제: {topic}")
    print(f"   - 시간 범위: {time_range}")
    print(f"   - 최대 결과: {max_results}개")
    print(f"   - 검색 깊이: {search_depth}")
    print(f"   - 출력 디렉토리: {output_dir}")
    print(f"\n📊 실행 순서:")
    print(f"   1️⃣  연구/뉴스 수집")
    print(f"   2️⃣  기술 트렌드 요약")
    print(f"   3️⃣  트렌드 예측")
    print(f"   4️⃣  리스크/기회 분석")
    print(f"   5️⃣  보고서 생성 (Markdown + PDF)")
    print("\n" + "="*80 + "\n")
    
    # 초기 State
    initial_state: TrendAnalysisState = {
        "topic": topic,
        "time_range": time_range,
        "max_results": max_results,
        "search_depth": search_depth
    }
    
    # 각 에이전트 설정
    config_research = ResearchNewsConfig(max_results=max_results or 20, search_depth=search_depth)
    config_tech_summary = TechTrendSummaryConfig(use_structured_output=True)
    config_prediction = TrendPredictionConfig(use_structured_output=True, temperature=0.3)
    config_risk_opp = RiskOpportunityConfig(use_structured_output=True)
    config_report = TrendReportConfig(output_dir=output_dir, generate_pdf=True)
    
    # 순차 실행 (LangGraph 버전 호환성 문제 해결)
    try:
        final_state = run_sequential_workflow(
            initial_state,
            config_research,
            config_tech_summary,
            config_prediction,
            config_risk_opp,
            config_report
        )
        
        print("\n" + "="*80)
        print("✅ 전체 워크플로우 완료!")
        print("="*80)
        
        # 최종 State 요약
        print(f"\n📊 최종 분석 결과:")
        print(f"   ✅ 수집된 자료: {len(final_state.get('research_news', {}).get('articles', []))}개")
        print(f"   ✅ 기술 카테고리: {len(final_state.get('tech_trend_summary', {}).get('technology_categories', []))}개")
        print(f"   ✅ 트렌드 예측: {len(final_state.get('trend_prediction', {}).get('predictions', []))}개")
        print(f"   ✅ 리스크/기회 분석: {len(final_state.get('risk_opportunity', {}).get('analyses', []))}개")
        
        print(f"\n🔍 실행 모드 요약:")
        print(f"   • 연구/뉴스 수집: {'🌐 Tavily API' if len(final_state.get('research_news', {}).get('sources', [])) > 1 else '⚠️ 레거시 더미 데이터'}")
        print(f"   • 기술 트렌드 요약: {'🤖 AI Structured Output' if len(final_state.get('tech_trend_summary', {}).get('technology_categories', [])) > 1 else '⚠️ 레거시 폴백'}")
        print(f"   • 트렌드 예측: {'🤖 AI Structured Output' if len(final_state.get('trend_prediction', {}).get('predictions', [])) > 0 else '⚠️ 레거시 폴백'}")
        print(f"   • 리스크/기회 분석: {'🤖 AI Structured Output' if len(final_state.get('risk_opportunity', {}).get('analyses', [])) > 0 else '⚠️ 레거시 폴백'}")
        
        # 보고서 저장
        report = final_state.get("trend_report", {})
        if report.get("markdown"):
            os.makedirs(output_dir, exist_ok=True)
            saved_path = save_report_markdown(
                report.get("markdown", ""),
                output_dir,
                report.get("filename", "trend_report.md")
            )
            print(f"\n💾 저장된 파일:")
            print(f"   📝 Markdown: {saved_path}")
            if report.get("pdf_path"):
                print(f"   📑 PDF: {report.get('pdf_path')}")
        
        print("\n" + "="*80)
        
        return final_state
        
    except Exception as e:
        print(f"\n❌ 워크플로우 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


# ----------------------------- CLI -----------------------------

def main():
    """CLI 실행"""
    parser = argparse.ArgumentParser(
        description="기술 트렌드 분석 AI 에이전트 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main_graph.py --topic "생성형 AI" --max-results 20
  python main_graph.py --topic "자율주행 기술" --depth basic
  python main_graph.py --topic "양자컴퓨팅" --time-range "2 years"
        """
    )
    
    parser.add_argument(
        "--topic",
        default="AI 기술 트렌드",
        help="분석할 기술 주제 (기본값: AI 기술 트렌드)"
    )
    parser.add_argument(
        "--time-range",
        default="1-2 years",
        help="분석 시간 범위 (기본값: 1-2 years)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=15,
        help="최대 수집 결과 수 (기본값: 15)"
    )
    parser.add_argument(
        "--depth",
        default="advanced",
        choices=["basic", "advanced"],
        help="검색 깊이 (기본값: advanced)"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="출력 디렉토리 (기본값: output)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 출력 모드"
    )
    
    args = parser.parse_args()
    
    # 환경 변수 확인
    tavily_key = os.getenv("TAVILY_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not tavily_key:
        print("⚠️  경고: TAVILY_API_KEY가 설정되지 않았습니다.")
        print("   뉴스 수집 기능이 제한될 수 있습니다.")
        print("   .env 파일에 TAVILY_API_KEY를 추가하세요.\n")
    
    if not openai_key:
        print("⚠️  경고: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   분석 및 예측 기능이 제한될 수 있습니다.")
        print("   .env 파일에 OPENAI_API_KEY를 추가하세요.\n")
    
    # 워크플로우 실행
    try:
        final_state = run_trend_analysis(
            topic=args.topic,
            time_range=args.time_range,
            max_results=args.max_results,
            search_depth=args.depth,
            output_dir=args.output_dir
        )
        
        # 상세 출력
        if args.verbose:
            print("\n" + "="*80)
            print("📊 상세 결과")
            print("="*80)
            
            if final_state.get("research_news"):
                print_research_news_results(final_state)
            
            if final_state.get("tech_trend_summary"):
                print_tech_trend_summary_results(final_state)
            
            if final_state.get("trend_prediction"):
                print_trend_prediction_results(final_state)
            
            if final_state.get("risk_opportunity"):
                print_risk_opportunity_results(final_state)
            
            if final_state.get("trend_report"):
                print_trend_report_summary(final_state)
        
        print("\n✅ 분석이 성공적으로 완료되었습니다!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 실행 실패: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

