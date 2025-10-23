"""
Module: agents/risk_opportunity_analysis.py
Purpose: 리스크 및 기회 분석 에이전트 (LangGraph 기반)

기능:
- 예상 트렌드에 따른 리스크 및 기회 요인 분석
- SWOT 분석 프레임워크 적용
- 리스크 우선순위화 및 완화 전략 제시
- 기회 포착 전략 제안

입력 State:
state = {
    "tech_trend_summary": {...},
    "trend_prediction": {
        "predictions": [...]
    }
}

출력 State:
state["risk_opportunity"] = {
    "analyses": [
        {
            "category": "생성형 AI",
            "opportunities": [
                {
                    "title": "콘텐츠 산업 혁신",
                    "description": "...",
                    "impact": "높음",
                    "timeframe": "단기",
                    "exploitation_strategy": "..."
                }
            ],
            "risks": [
                {
                    "title": "저작권 및 윤리 문제",
                    "description": "...",
                    "severity": "높음",
                    "likelihood": "높음",
                    "mitigation_strategy": "..."
                }
            ],
            "swot": {
                "strengths": [...],
                "weaknesses": [...],
                "opportunities": [...],
                "threats": [...]
            }
        }
    ],
    "summary": "전체 리스크/기회 요약"
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
    class Opportunity(BaseModel):
        """기회 요인"""
        title: str = Field(description="기회 제목")
        description: str = Field(description="기회 설명")
        impact: str = Field(description="영향도 (낮음|보통|높음|매우높음)")
        timeframe: str = Field(description="실현 시기 (단기|중기|장기)")
        exploitation_strategy: str = Field(description="기회 활용 전략")
        
        model_config = ConfigDict(str_strip_whitespace=True)
    
    class Risk(BaseModel):
        """리스크 요인"""
        title: str = Field(description="리스크 제목")
        description: str = Field(description="리스크 설명")
        severity: str = Field(description="심각도 (낮음|보통|높음|매우높음)")
        likelihood: str = Field(description="발생 가능성 (낮음|보통|높음)")
        mitigation_strategy: str = Field(description="완화 전략")
        
        model_config = ConfigDict(str_strip_whitespace=True)
    
    class SWOT(BaseModel):
        """SWOT 분석"""
        strengths: List[str] = Field(default=[], description="강점")
        weaknesses: List[str] = Field(default=[], description="약점")
        opportunities: List[str] = Field(default=[], description="기회")
        threats: List[str] = Field(default=[], description="위협")
        
        model_config = ConfigDict(str_strip_whitespace=True)
    
    class RiskOpportunityAnalysis(BaseModel):
        """기술별 리스크/기회 분석"""
        category: str = Field(description="기술 카테고리")
        opportunities: List[Opportunity] = []
        risks: List[Risk] = []
        swot: SWOT
        
        model_config = ConfigDict(str_strip_whitespace=True)
    
    class RiskOpportunityResult(BaseModel):
        """전체 리스크/기회 분석 결과"""
        analyses: List[RiskOpportunityAnalysis]
        summary: str = Field(description="전체 요약")
else:
    Opportunity = dict
    Risk = dict
    SWOT = dict
    RiskOpportunityAnalysis = dict
    RiskOpportunityResult = dict


# ----------------------------- State & Config -----------------------------

class RiskOpportunityOutput(TypedDict, total=False):
    analyses: List[Dict[str, Any]]
    summary: str

class RiskOpportunityState(TypedDict, total=False):
    tech_trend_summary: Dict[str, Any]
    trend_prediction: Dict[str, Any]
    risk_opportunity: RiskOpportunityOutput

@dataclass
class RiskOpportunityConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    use_structured_output: bool = True


# ----------------------------- Utilities -----------------------------

def _norm(text: str) -> str:
    return (text or "").strip()


# ----------------------------- Prompts (외부 파일에서 로드) -----------------------------

try:
    from prompts.risk_opportunity_prompts import (
        RISK_OPPORTUNITY_SYSTEM_PROMPT,
        RISK_OPPORTUNITY_USER_QUERY_TMPL
    )
except ImportError:
    # 폴백: 프롬프트를 찾을 수 없는 경우 기본값 사용
    RISK_OPPORTUNITY_SYSTEM_PROMPT = """당신은 기술 트렌드 리스크 및 기회 분석 전문가입니다.

**분석 목표: 예상 트렌드에 따른 리스크 요인과 기회 요인을 체계적으로 분석**

핵심 원칙:
1. **균형적 시각**: 낙관론과 비관론의 균형, 현실적 분석
2. **구체성**: 추상적 표현 지양, 구체적 사례와 근거 제시
3. **실행 가능성**: 실제 활용 가능한 전략 제안
4. **우선순위화**: 영향도와 가능성 기반 우선순위 설정

분석 프레임워크:

1. **기회 요인 (Opportunities)**
   각 기회마다 다음 정보 포함:
   - title: 기회의 명확한 제목
   - description: 왜 기회인지, 어떤 가치를 창출하는지
   - impact: 영향도 (낮음|보통|높음|매우높음)
   - timeframe: 실현 시기 (단기|중기|장기)
   - exploitation_strategy: 기회를 활용하기 위한 구체적 전략

   예시:
   {{
     "title": "금융 산업 AI 도입 가속화",
     "description": "규제 완화와 기술 성숙으로 금융권 AI 투자 증가 예상",
     "impact": "높음",
     "timeframe": "단기",
     "exploitation_strategy": "금융 특화 AI 솔루션 개발 및 파트너십 확대"
   }}

2. **리스크 요인 (Risks)**
   각 리스크마다 다음 정보 포함:
   - title: 리스크의 명확한 제목
   - description: 리스크의 내용과 영향
   - severity: 심각도 (낮음|보통|높음|매우높음)
   - likelihood: 발생 가능성 (낮음|보통|높음)
   - mitigation_strategy: 리스크 완화를 위한 구체적 전략

   예시:
   {{
     "title": "AI 규제 강화",
     "description": "EU AI Act 등 규제로 인한 개발 및 배포 제약",
     "severity": "높음",
     "likelihood": "높음",
     "mitigation_strategy": "규제 준수 프레임워크 구축 및 설명가능 AI 기술 개발"
   }}

3. **SWOT 분석**
   - strengths: 기술의 내부 강점 (2-5개)
   - weaknesses: 기술의 내부 약점 (2-5개)
   - opportunities: 외부 기회 요인 (2-5개)
   - threats: 외부 위협 요인 (2-5개)

4. **리스크 매트릭스**
   - 심각도 × 가능성으로 우선순위 결정
   - 높음×높음 > 높음×보통 > ... 순으로 중요도 평가

출력 스키마 (JSON만):
{{
  "analyses": [
    {{
      "category": "",
      "opportunities": [{{...}}],
      "risks": [{{...}}],
      "swot": {{
        "strengths": [],
        "weaknesses": [],
        "opportunities": [],
        "threats": []
      }}
    }}
  ],
  "summary": "전체 리스크/기회 분석을 3-5문장으로 요약"
}}
"""
    RISK_OPPORTUNITY_USER_QUERY_TMPL = """다음 기술 트렌드 및 예측을 바탕으로 리스크와 기회를 분석하세요.

=== 기술 트렌드 요약 ===
{trend_summary}

=== 트렌드 예측 ===
{trend_prediction}
===========================

**분석 지침:**
1. 각 기술 카테고리마다 최소 2-3개의 기회와 리스크 식별
2. 기회는 실현 가능성이 높은 것부터, 리스크는 영향도가 큰 것부터 우선순위화
3. 각 기회/리스크마다 구체적인 전략 제시
4. SWOT 분석으로 종합적 시각 제공
5. 전체 요약에서 핵심 메시지 전달

JSON으로만 반환하세요.
"""


# ----------------------------- Node Factory -----------------------------

def _risk_opportunity_node_factory(cfg: RiskOpportunityConfig):
    """LangGraph 노드 함수 팩토리"""
    
    def node(state: RiskOpportunityState) -> RiskOpportunityState:
        tech_summary = state.get("tech_trend_summary", {})
        trend_pred = state.get("trend_prediction", {})
        
        categories = tech_summary.get("technology_categories", [])
        predictions = trend_pred.get("predictions", [])
        
        print(f"\n⚖️  리스크 및 기회 분석 시작")
        print(f"   - 분석 대상: {len(categories)}개 기술")
        
        if not categories and not predictions:
            print(f"⚠️ 분석할 데이터가 없습니다.")
            state["risk_opportunity"] = {
                "analyses": [],
                "summary": "분석할 트렌드 데이터가 부족합니다."
            }
            return state
        
        # 입력 데이터 텍스트화
        trend_summary_text = json.dumps(categories, ensure_ascii=False, indent=2)
        trend_prediction_text = json.dumps(predictions, ensure_ascii=False, indent=2)
        
        # LLM Structured Output 방식
        if cfg.use_structured_output and LANGCHAIN_AVAILABLE and PYDANTIC_AVAILABLE:
            try:
                llm = ChatOpenAI(model=cfg.model, temperature=cfg.temperature)
                structured_llm = llm.with_structured_output(RiskOpportunityResult)
                
                sys_prompt = RISK_OPPORTUNITY_SYSTEM_PROMPT
                user_prompt = RISK_OPPORTUNITY_USER_QUERY_TMPL.format(
                    trend_summary=trend_summary_text,
                    trend_prediction=trend_prediction_text
                )
                
                print(f"   🤖 AI Structured Output 분석 중...")
                result: RiskOpportunityResult = structured_llm.invoke([
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content=user_prompt)
                ])
                
                print(f"   ✅ AI 분석 완료 (Structured Output)")
                
                # Pydantic → Dict 변환
                analyses = []
                for analysis in result.analyses:
                    opportunities = []
                    for opp in analysis.opportunities:
                        opportunities.append({
                            "title": opp.title,
                            "description": opp.description,
                            "impact": opp.impact,
                            "timeframe": opp.timeframe,
                            "exploitation_strategy": opp.exploitation_strategy
                        })
                    
                    risks = []
                    for risk in analysis.risks:
                        risks.append({
                            "title": risk.title,
                            "description": risk.description,
                            "severity": risk.severity,
                            "likelihood": risk.likelihood,
                            "mitigation_strategy": risk.mitigation_strategy
                        })
                    
                    analyses.append({
                        "category": analysis.category,
                        "opportunities": opportunities,
                        "risks": risks,
                        "swot": {
                            "strengths": analysis.swot.strengths,
                            "weaknesses": analysis.swot.weaknesses,
                            "opportunities": analysis.swot.opportunities,
                            "threats": analysis.swot.threats
                        }
                    })
                
                state["risk_opportunity"] = {
                    "analyses": analyses,
                    "summary": result.summary
                }
                
            except Exception as e:
                print(f"   ⚠️ Structured Output 실패, 레거시 방식 사용")
                print(f"   오류: {str(e)[:100]}")
                # 폴백
                state["risk_opportunity"] = _fallback_risk_opportunity(categories)
        else:
            # 기본 방식
            print(f"   ⚠️ 레거시 분석 모드 (Structured Output 비활성화)")
            state["risk_opportunity"] = _fallback_risk_opportunity(categories)
        
        # 결과 출력 (간소화)
        ro_analysis = state["risk_opportunity"]
        
        total_opportunities = sum(len(a.get("opportunities", [])) for a in ro_analysis.get("analyses", []))
        total_risks = sum(len(a.get("risks", [])) for a in ro_analysis.get("analyses", []))
        
        print(f"\n   ✅ 분석 완료: {len(ro_analysis.get('analyses', []))}개 기술")
        print(f"   💡 기회: {total_opportunities}개 | ⚠️  리스크: {total_risks}개")
        
        for idx, analysis in enumerate(ro_analysis.get("analyses", [])[:2], 1):
            opps = analysis.get("opportunities", [])
            risks = analysis.get("risks", [])
            print(f"   [{idx}] {analysis.get('category', 'N/A')} - 기회 {len(opps)}/리스크 {len(risks)}")
        
        if len(ro_analysis.get("analyses", [])) > 2:
            print(f"   ... 외 {len(ro_analysis['analyses'])-2}개")
        
        return state
    
    return node


def _fallback_risk_opportunity(categories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """폴백: 간단한 리스크/기회 분석"""
    analyses = []
    
    for cat in categories[:3]:
        category_name = cat.get("category", "기술")
        analyses.append({
            "category": category_name,
            "opportunities": [
                {
                    "title": f"{category_name} 시장 성장",
                    "description": "기술 발전으로 인한 시장 확대 기회",
                    "impact": "높음",
                    "timeframe": "중기",
                    "exploitation_strategy": "선제적 기술 투자 및 시장 진입"
                }
            ],
            "risks": [
                {
                    "title": "기술 경쟁 심화",
                    "description": "다수 기업의 시장 진입으로 경쟁 격화",
                    "severity": "보통",
                    "likelihood": "높음",
                    "mitigation_strategy": "차별화된 기술 역량 확보"
                }
            ],
            "swot": {
                "strengths": ["기술 혁신성", "시장 선점"],
                "weaknesses": ["높은 개발 비용", "기술 리스크"],
                "opportunities": ["시장 성장", "규제 완화"],
                "threats": ["경쟁 심화", "규제 불확실성"]
            }
        })
    
    return {
        "analyses": analyses,
        "summary": f"{len(categories)}개 기술에 대한 리스크/기회 분석을 완료했습니다."
    }


# ----------------------------- Graph Builder -----------------------------

def build_risk_opportunity_graph(config: Optional[RiskOpportunityConfig] = None):
    cfg = config or RiskOpportunityConfig()
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph가 설치되어 있지 않습니다. `pip install langgraph` 후 다시 시도하세요.")
    
    g = StateGraph(RiskOpportunityState)
    g.add_node("risk_opportunity_analysis", _risk_opportunity_node_factory(cfg))
    g.add_edge(START, "risk_opportunity_analysis")
    g.add_edge("risk_opportunity_analysis", END)
    return g.compile()


# ----------------------------- Helper -----------------------------

def run_risk_opportunity_analysis(state: Dict[str, Any], config: Optional[RiskOpportunityConfig] = None) -> Dict[str, Any]:
    """리스크/기회 분석 실행"""
    app = build_risk_opportunity_graph(config)
    return app.invoke(state)


# ----------------------------- Output Formatting -----------------------------

def print_risk_opportunity_results(result: Dict[str, Any]):
    """분석 결과를 보기 좋게 출력"""
    ro_analysis = result.get("risk_opportunity", {})
    analyses = ro_analysis.get("analyses", [])
    
    print("\n" + "=" * 80)
    print("⚖️  리스크 및 기회 분석 결과")
    print("=" * 80)
    
    print(f"\n📝 종합 요약:")
    print(f"{ro_analysis.get('summary', 'N/A')}")
    
    print(f"\n📊 기술별 분석 ({len(analyses)}개):")
    print("-" * 80)
    
    for idx, analysis in enumerate(analyses, 1):
        print(f"\n[{idx}] {analysis.get('category', 'Unknown')}")
        
        # 기회
        opportunities = analysis.get("opportunities", [])
        if opportunities:
            print(f"\n    💡 기회 요인 ({len(opportunities)}개):")
            for opp_idx, opp in enumerate(opportunities, 1):
                print(f"       [{opp_idx}] {opp.get('title', 'N/A')}")
                print(f"           {opp.get('description', '')}")
                print(f"           영향도: {opp.get('impact', 'N/A')} | 시기: {opp.get('timeframe', 'N/A')}")
                print(f"           전략: {opp.get('exploitation_strategy', 'N/A')}")
        
        # 리스크
        risks = analysis.get("risks", [])
        if risks:
            print(f"\n    ⚠️  리스크 요인 ({len(risks)}개):")
            for risk_idx, risk in enumerate(risks, 1):
                print(f"       [{risk_idx}] {risk.get('title', 'N/A')}")
                print(f"           {risk.get('description', '')}")
                print(f"           심각도: {risk.get('severity', 'N/A')} | 가능성: {risk.get('likelihood', 'N/A')}")
                print(f"           완화 전략: {risk.get('mitigation_strategy', 'N/A')}")
        
        # SWOT
        swot = analysis.get("swot", {})
        if swot:
            print(f"\n    📊 SWOT 분석:")
            if swot.get("strengths"):
                print(f"       💪 강점: {', '.join(swot['strengths'])}")
            if swot.get("weaknesses"):
                print(f"       ⚖️  약점: {', '.join(swot['weaknesses'])}")
            if swot.get("opportunities"):
                print(f"       🎯 기회: {', '.join(swot['opportunities'])}")
            if swot.get("threats"):
                print(f"       ⚠️  위협: {', '.join(swot['threats'])}")
    
    print("\n" + "=" * 80)


# ----------------------------- CLI Test -----------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("⚖️  리스크 및 기회 분석 에이전트 테스트")
    print("=" * 80)
    
    # 모의 데이터
    dummy_state = {
        "tech_trend_summary": {
            "technology_categories": [
                {
                    "category": "생성형 AI",
                    "key_technologies": ["GPT-4", "Claude"],
                    "summary": "빠른 발전과 상용화",
                    "maturity_level": "성숙기",
                    "trend_direction": "상승"
                }
            ]
        },
        "trend_prediction": {
            "predictions": [
                {
                    "category": "생성형 AI",
                    "short_term": {
                        "prediction": "기업 도입 가속화",
                        "confidence": 0.8
                    },
                    "market_impact": "매우높음",
                    "adoption_curve": "초기 대중화 단계"
                }
            ]
        }
    }
    
    try:
        config = RiskOpportunityConfig()
        final = run_risk_opportunity_analysis(dummy_state, config)
        print_risk_opportunity_results(final)
        
        # JSON 저장
        output_file = "risk_opportunity_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final.get("risk_opportunity"), f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과가 저장되었습니다: {output_file}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

