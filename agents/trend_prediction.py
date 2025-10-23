"""
Module: agents/trend_prediction.py
Purpose: 트렌드 예측 에이전트 (LangGraph 기반)

기능:
- 각 기술 트렌드의 발전 방향 및 시장 적용 가능성 예측
- 단기(1년), 중기(3년), 장기(5년) 예측
- 기술 성숙도 곡선 분석
- 시장 영향도 평가

입력 State:
state = {
    "tech_trend_summary": {
        "technology_categories": [...],
        "overall_summary": "..."
    }
}

출력 State:
state["trend_prediction"] = {
    "predictions": [
        {
            "category": "생성형 AI",
            "short_term": {"timeframe": "1년", "prediction": "...", "confidence": 0.85},
            "mid_term": {"timeframe": "3년", "prediction": "...", "confidence": 0.70},
            "long_term": {"timeframe": "5년", "prediction": "...", "confidence": 0.55},
            "market_impact": "높음",
            "adoption_curve": "초기 대중화 단계",
            "key_drivers": ["기술 발전", "시장 수요"],
            "barriers": ["규제", "비용"]
        },
        ...
    ],
    "overall_outlook": "..."
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
    class TimeframePrediction(BaseModel):
        """시간대별 예측"""
        timeframe: str = Field(description="기간 (예: 1년, 3년)")
        prediction: str = Field(description="예측 내용")
        confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="신뢰도")
        
        model_config = ConfigDict(str_strip_whitespace=True)
    
    class TechnologyPrediction(BaseModel):
        """기술별 예측"""
        category: str = Field(description="기술 카테고리")
        short_term: TimeframePrediction
        mid_term: TimeframePrediction
        long_term: TimeframePrediction
        market_impact: str = Field(description="시장 영향도 (낮음|보통|높음|매우높음)")
        adoption_curve: str = Field(description="기술 채택 곡선 위치")
        key_drivers: List[str] = Field(default=[], description="주요 성장 동력")
        barriers: List[str] = Field(default=[], description="장애 요인")
        
        model_config = ConfigDict(str_strip_whitespace=True)
    
    class TrendPredictionResult(BaseModel):
        """트렌드 예측 결과"""
        predictions: List[TechnologyPrediction]
        overall_outlook: str = Field(description="전체 전망")
else:
    TimeframePrediction = dict
    TechnologyPrediction = dict
    TrendPredictionResult = dict


# ----------------------------- State & Config -----------------------------

class TrendPredictionOutput(TypedDict, total=False):
    predictions: List[Dict[str, Any]]
    overall_outlook: str

class TrendPredictionState(TypedDict, total=False):
    tech_trend_summary: Dict[str, Any]
    trend_prediction: TrendPredictionOutput

@dataclass
class TrendPredictionConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    use_structured_output: bool = True


# ----------------------------- Utilities -----------------------------

def _norm(text: str) -> str:
    return (text or "").strip()


# ----------------------------- Prompts (외부 파일에서 로드) -----------------------------

try:
    from prompts.trend_prediction_prompts import (
        TREND_PREDICTION_SYSTEM_PROMPT,
        TREND_PREDICTION_USER_QUERY_TMPL
    )
except ImportError:
    # 폴백: 프롬프트를 찾을 수 없는 경우 기본값 사용
    TREND_PREDICTION_SYSTEM_PROMPT = """당신은 기술 트렌드 예측 전문가입니다.

**분석 목표: 기술 트렌드의 발전 방향 및 시장 적용 가능성 예측**

핵심 원칙:
1. **시간대별 예측**: 단기(1년), 중기(3년), 장기(5년)으로 구분하여 예측
2. **신뢰도 평가**: 각 예측의 신뢰도를 0.0~1.0으로 수치화
3. **시장 영향도**: 해당 기술이 시장에 미칠 영향 평가
4. **현실적 접근**: 과도한 낙관론이나 비관론 지양

분석 작업:

1. **단기 예측 (short_term: 1년)**
   - 현재 기술 수준에서 1년 내 예상되는 발전
   - 이미 진행 중인 프로젝트나 계획 중심
   - 신뢰도: 0.7~0.9 (비교적 높음)

2. **중기 예측 (mid_term: 3년)**
   - 3년 후 기술의 성숙도와 시장 상황
   - 상용화 가능성, 표준화 진행 등
   - 신뢰도: 0.5~0.7 (중간)

3. **장기 예측 (long_term: 5년)**
   - 5년 후 기술의 진화 방향
   - 새로운 패러다임이나 응용 분야
   - 신뢰도: 0.3~0.6 (낮음)

4. **시장 영향도 (market_impact)**
   - "낮음": 니치 시장, 제한적 영향
   - "보통": 특정 산업에 유의미한 영향
   - "높음": 여러 산업에 광범위한 영향
   - "매우높음": 산업 전반의 패러다임 변화

5. **기술 채택 곡선 (adoption_curve)**
   - "혁신자 단계" (2.5%)
   - "얼리어답터 단계" (13.5%)
   - "초기 대중화 단계" (34%)
   - "후기 대중화 단계" (34%)
   - "지각 수용자 단계" (16%)

6. **성장 동력 (key_drivers)**
   - 기술 발전을 촉진하는 요인들
   - 예: ["시장 수요 증가", "정부 지원", "비용 절감"]

7. **장애 요인 (barriers)**
   - 기술 확산을 저해하는 요소들
   - 예: ["규제", "높은 초기 비용", "기술적 한계"]

출력 스키마 (JSON만):
{{
  "predictions": [
    {{
      "category": "",
      "short_term": {{"timeframe": "1년", "prediction": "", "confidence": 0.8}},
      "mid_term": {{"timeframe": "3년", "prediction": "", "confidence": 0.6}},
      "long_term": {{"timeframe": "5년", "prediction": "", "confidence": 0.5}},
      "market_impact": "",
      "adoption_curve": "",
      "key_drivers": [],
      "barriers": []
    }}
  ],
  "overall_outlook": "전체 기술 트렌드 전망을 3-5문장으로 요약"
}}
"""
    TREND_PREDICTION_USER_QUERY_TMPL = """다음 기술 트렌드 요약을 바탕으로 발전 방향과 시장 적용 가능성을 예측하세요.

=== 기술 트렌드 요약 ===
{summary_text}
=========================

**예측 지침:**
1. 각 기술 카테고리마다 단기/중기/장기 예측 제시
2. 예측의 신뢰도를 현실적으로 평가 (단기 > 중기 > 장기)
3. 시장 영향도와 기술 채택 단계 명시
4. 성장 동력과 장애 요인을 구체적으로 나열
5. 전체 전망에서 핵심 메시지 전달

JSON으로만 반환하세요.
"""


# ----------------------------- Node Factory -----------------------------

def _trend_prediction_node_factory(cfg: TrendPredictionConfig):
    """LangGraph 노드 함수 팩토리"""
    
    def node(state: TrendPredictionState) -> TrendPredictionState:
        tech_summary = state.get("tech_trend_summary", {})
        categories = tech_summary.get("technology_categories", [])
        overall_summary = tech_summary.get("overall_summary", "")
        
        print(f"\n🔮 트렌드 예측 시작")
        print(f"   - 분석 대상: {len(categories)}개 기술 카테고리")
        
        if not categories:
            print(f"⚠️ 예측할 기술 카테고리가 없습니다.")
            state["trend_prediction"] = {
                "predictions": [],
                "overall_outlook": "분석할 기술 트렌드가 부족합니다."
            }
            return state
        
        # 요약 텍스트 생성
        summary_lines = [f"전체 요약: {overall_summary}\n"]
        for idx, cat in enumerate(categories, 1):
            summary_lines.append(f"[{idx}] {cat.get('category', 'N/A')}")
            summary_lines.append(f"   - 주요 기술: {', '.join(cat.get('key_technologies', []))}")
            summary_lines.append(f"   - 요약: {cat.get('summary', '')}")
            summary_lines.append(f"   - 성숙도: {cat.get('maturity_level', 'N/A')}, 트렌드: {cat.get('trend_direction', 'N/A')}")
            summary_lines.append("")
        
        summary_text = "\n".join(summary_lines)
        
        # LLM Structured Output 방식
        if cfg.use_structured_output and LANGCHAIN_AVAILABLE and PYDANTIC_AVAILABLE:
            try:
                llm = ChatOpenAI(model=cfg.model, temperature=cfg.temperature)
                structured_llm = llm.with_structured_output(TrendPredictionResult)
                
                sys_prompt = TREND_PREDICTION_SYSTEM_PROMPT
                user_prompt = TREND_PREDICTION_USER_QUERY_TMPL.format(
                    summary_text=summary_text
                )
                
                print(f"   🤖 AI Structured Output 예측 중...")
                result: TrendPredictionResult = structured_llm.invoke([
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content=user_prompt)
                ])
                
                print(f"   ✅ AI 예측 완료 (Structured Output)")
                
                # Pydantic → Dict 변환
                predictions = []
                for pred in result.predictions:
                    predictions.append({
                        "category": pred.category,
                        "short_term": {
                            "timeframe": pred.short_term.timeframe,
                            "prediction": pred.short_term.prediction,
                            "confidence": pred.short_term.confidence
                        },
                        "mid_term": {
                            "timeframe": pred.mid_term.timeframe,
                            "prediction": pred.mid_term.prediction,
                            "confidence": pred.mid_term.confidence
                        },
                        "long_term": {
                            "timeframe": pred.long_term.timeframe,
                            "prediction": pred.long_term.prediction,
                            "confidence": pred.long_term.confidence
                        },
                        "market_impact": pred.market_impact,
                        "adoption_curve": pred.adoption_curve,
                        "key_drivers": pred.key_drivers,
                        "barriers": pred.barriers
                    })
                
                state["trend_prediction"] = {
                    "predictions": predictions,
                    "overall_outlook": result.overall_outlook
                }
                
            except Exception as e:
                print(f"   ⚠️ Structured Output 실패, 레거시 방식 사용")
                print(f"   오류: {str(e)[:100]}")
                # 폴백
                state["trend_prediction"] = _fallback_prediction(categories)
        else:
            # 기본 방식
            print(f"   ⚠️ 레거시 예측 모드 (Structured Output 비활성화)")
            state["trend_prediction"] = _fallback_prediction(categories)
        
        # 결과 출력 (간소화)
        prediction = state["trend_prediction"]
        
        print(f"\n   ✅ 예측 완료: {len(prediction.get('predictions', []))}개 기술")
        
        for idx, pred in enumerate(prediction.get("predictions", [])[:2], 1):
            short = pred.get("short_term", {})
            print(f"   [{idx}] {pred.get('category', 'N/A')} - {pred.get('market_impact', 'N/A')} 영향 (신뢰도: {short.get('confidence', 0):.0%})")
        
        if len(prediction.get("predictions", [])) > 2:
            print(f"   ... 외 {len(prediction['predictions'])-2}개")
        
        return state
    
    return node


def _fallback_prediction(categories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """폴백: 간단한 예측"""
    predictions = []
    
    for cat in categories[:3]:
        category_name = cat.get("category", "기술")
        predictions.append({
            "category": category_name,
            "short_term": {
                "timeframe": "1년",
                "prediction": f"{category_name} 기술이 점진적으로 발전할 것으로 예상됩니다.",
                "confidence": 0.7
            },
            "mid_term": {
                "timeframe": "3년",
                "prediction": f"{category_name} 기술이 주요 산업에 적용될 것으로 보입니다.",
                "confidence": 0.5
            },
            "long_term": {
                "timeframe": "5년",
                "prediction": f"{category_name} 기술이 성숙 단계에 도달할 가능성이 있습니다.",
                "confidence": 0.4
            },
            "market_impact": "보통",
            "adoption_curve": "초기 대중화 단계",
            "key_drivers": ["기술 발전", "시장 수요"],
            "barriers": ["규제", "비용"]
        })
    
    return {
        "predictions": predictions,
        "overall_outlook": f"{len(categories)}개 기술 트렌드에 대한 예측을 완료했습니다."
    }


# ----------------------------- Graph Builder -----------------------------

def build_trend_prediction_graph(config: Optional[TrendPredictionConfig] = None):
    cfg = config or TrendPredictionConfig()
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph가 설치되어 있지 않습니다. `pip install langgraph` 후 다시 시도하세요.")
    
    g = StateGraph(TrendPredictionState)
    g.add_node("trend_prediction", _trend_prediction_node_factory(cfg))
    g.add_edge(START, "trend_prediction")
    g.add_edge("trend_prediction", END)
    return g.compile()


# ----------------------------- Helper -----------------------------

def run_trend_prediction(state: Dict[str, Any], config: Optional[TrendPredictionConfig] = None) -> Dict[str, Any]:
    """트렌드 예측 실행"""
    app = build_trend_prediction_graph(config)
    return app.invoke(state)


# ----------------------------- Output Formatting -----------------------------

def print_trend_prediction_results(result: Dict[str, Any]):
    """예측 결과를 보기 좋게 출력"""
    prediction = result.get("trend_prediction", {})
    predictions = prediction.get("predictions", [])
    
    print("\n" + "=" * 80)
    print("🔮 트렌드 예측 결과")
    print("=" * 80)
    
    print(f"\n🌐 전체 전망:")
    print(f"{prediction.get('overall_outlook', 'N/A')}")
    
    print(f"\n📊 기술별 예측 ({len(predictions)}개):")
    print("-" * 80)
    
    for idx, pred in enumerate(predictions, 1):
        print(f"\n[{idx}] {pred.get('category', 'Unknown')}")
        print(f"    📈 시장 영향도: {pred.get('market_impact', 'N/A')}")
        print(f"    📊 기술 채택 곡선: {pred.get('adoption_curve', 'N/A')}")
        
        short = pred.get("short_term", {})
        mid = pred.get("mid_term", {})
        long = pred.get("long_term", {})
        
        print(f"\n    🔹 단기 예측 ({short.get('timeframe', '1년')}) [신뢰도: {short.get('confidence', 0):.0%}]")
        print(f"       {short.get('prediction', 'N/A')}")
        
        print(f"\n    🔸 중기 예측 ({mid.get('timeframe', '3년')}) [신뢰도: {mid.get('confidence', 0):.0%}]")
        print(f"       {mid.get('prediction', 'N/A')}")
        
        print(f"\n    🔺 장기 예측 ({long.get('timeframe', '5년')}) [신뢰도: {long.get('confidence', 0):.0%}]")
        print(f"       {long.get('prediction', 'N/A')}")
        
        drivers = pred.get("key_drivers", [])
        if drivers:
            print(f"\n    ✅ 성장 동력: {', '.join(drivers)}")
        
        barriers = pred.get("barriers", [])
        if barriers:
            print(f"    ⚠️  장애 요인: {', '.join(barriers)}")
    
    print("\n" + "=" * 80)


# ----------------------------- CLI Test -----------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("🔮 트렌드 예측 에이전트 테스트")
    print("=" * 80)
    
    # 모의 데이터
    dummy_state = {
        "tech_trend_summary": {
            "technology_categories": [
                {
                    "category": "생성형 AI",
                    "key_technologies": ["GPT-4", "Claude", "Gemini"],
                    "summary": "대규모 언어 모델이 빠르게 발전하며 다양한 산업에 적용되고 있습니다.",
                    "maturity_level": "성숙기",
                    "trend_direction": "상승"
                },
                {
                    "category": "자율주행",
                    "key_technologies": ["Tesla FSD", "Waymo", "Cruise"],
                    "summary": "레벨 4 자율주행 기술이 특정 지역에서 상용화되기 시작했습니다.",
                    "maturity_level": "발전 중",
                    "trend_direction": "상승"
                }
            ],
            "overall_summary": "AI 기술 전반에서 빠른 발전이 이루어지고 있습니다."
        }
    }
    
    try:
        config = TrendPredictionConfig()
        final = run_trend_prediction(dummy_state, config)
        print_trend_prediction_results(final)
        
        # JSON 저장
        output_file = "trend_prediction_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final.get("trend_prediction"), f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과가 저장되었습니다: {output_file}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

