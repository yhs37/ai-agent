"""
Module: agents/trend_report_generator.py
Purpose: 트렌드 보고서 작성 에이전트 (LangGraph 기반)

기능:
- 트렌드 분석 결과를 정리한 종합 보고서 작성
- Markdown 및 PDF 형식 지원
- 전문적인 보고서 구조 및 디자인
- 챗, 표, 그래프 등 시각화 요소 포함

입력 State:
state = {
    "research_news": {...},
    "tech_trend_summary": {...},
    "trend_prediction": {...},
    "risk_opportunity": {...}
}

출력 State:
state["trend_report"] = {
    "title": "AI 기술 트렌드 분석 보고서",
    "markdown": "# ...",
    "filename": "trend_report_2025_01_20.md",
    "pdf_path": "outputs/trend_report_2025_01_20.pdf"
}

환경 변수:
- OPENAI_API_KEY (선택 - 요약 개선용)
"""
from __future__ import annotations

import os
import json
from datetime import datetime
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
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# PDF 생성을 위한 reportlab
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False
    print("⚠️ reportlab이 설치되어 있지 않습니다. PDF 생성 기능이 비활성화됩니다.")


# ----------------------------- State & Config -----------------------------

class TrendReportOutput(TypedDict, total=False):
    title: str
    markdown: str
    filename: str
    pdf_path: str

class TrendReportState(TypedDict, total=False):
    research_news: Dict[str, Any]
    tech_trend_summary: Dict[str, Any]
    trend_prediction: Dict[str, Any]
    risk_opportunity: Dict[str, Any]
    trend_report: TrendReportOutput

@dataclass
class TrendReportConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    use_llm_summary: bool = False  # LLM으로 요약 개선
    output_dir: str = "output"
    generate_pdf: bool = True  # PDF 생성


# ----------------------------- Utilities -----------------------------

def _norm(text: str) -> str:
    return (text or "").strip()

def _fmt_list(items: List[str], max_items: int = 5) -> str:
    """리스트를 문자열로 포맷"""
    if not items:
        return "없음"
    display_items = items[:max_items]
    result = ", ".join(display_items)
    if len(items) > max_items:
        result += f" 외 {len(items)-max_items}개"
    return result


# ----------------------------- Report Template -----------------------------

def _generate_markdown_report(state: TrendReportState, timestamp: str) -> str:
    """Markdown 보고서 생성"""
    
    research_news = state.get("research_news", {})
    tech_summary = state.get("tech_trend_summary", {})
    predictions = state.get("trend_prediction", {})
    ro_analysis = state.get("risk_opportunity", {})
    
    # 기본 정보
    topic = research_news.get("query", "기술 트렌드")
    collection_date = research_news.get("collection_date", "N/A")
    articles = research_news.get("articles", [])
    
    # 보고서 시작
    md = f"""# 🌐 {topic} 트렌드 분석 보고서

_작성 시점: {timestamp}_  
_분석 기간: {collection_date}_

---

## 📋 목차

1. [요약](#요약)
2. [연구 및 뉴스 수집 결과](#연구-및-뉴스-수집-결과)
3. [기술 트렌드 요약](#기술-트렌드-요약)
4. [트렌드 예측](#트렌드-예측)
5. [리스크 및 기회 분석](#리스크-및-기회-분석)
6. [결론 및 제언](#결론-및-제언)
7. [참고 자료](#참고-자료)

---

## 📊 요약

"""
    
    # 1. 요약 (Executive Summary)
    tech_categories = tech_summary.get("technology_categories", [])
    pred_list = predictions.get("predictions", [])
    analyses = ro_analysis.get("analyses", [])
    
    md += f"""### 핵심 발견사항

- **수집 자료**: {len(articles)}개의 연구/뉴스 기사 분석
- **주요 기술**: {len(tech_categories)}개 카테고리 식별
- **예측 범위**: 단기(1년), 중기(3년), 장기(5년) 전망
- **리스크/기회**: {len(analyses)}개 기술에 대한 SWOT 분석 완료

### 주요 인사이트

"""
    
    # 핵심 인사이트
    key_insights = tech_summary.get("key_insights", [])
    if key_insights:
        for idx, insight in enumerate(key_insights[:5], 1):
            md += f"{idx}. {insight}\n"
    else:
        md += f"- {topic} 분야에서 빠른 기술 발전이 진행 중입니다.\n"
    
    md += "\n### 전체 전망\n\n"
    md += predictions.get("overall_outlook", "기술 트렌드가 긍정적으로 발전할 것으로 예상됩니다.") + "\n\n"
    
    md += "---\n\n"
    
    # 2. 연구 및 뉴스 수집 결과
    md += """## 📰 연구 및 뉴스 수집 결과

### 수집 개요

"""
    
    research_count = sum(1 for a in articles if a.get("category") == "research")
    news_count = len(articles) - research_count
    sources = research_news.get("sources", [])
    
    md += f"- **총 자료 수**: {len(articles)}개\n"
    md += f"- **연구 자료**: {research_count}개\n"
    md += f"- **뉴스 기사**: {news_count}개\n"
    md += f"- **출처**: {len(sources)}개 ({_fmt_list(sources, 10)})\n\n"
    
    md += "### 주요 자료\n\n"
    
    for idx, article in enumerate(articles[:10], 1):
        title = article.get("title", "N/A")
        url = article.get("url", "#")
        category = article.get("category", "news")
        date = article.get("published_date", "N/A")
        relevance = article.get("relevance_score", 0)
        
        category_emoji = "📄" if category == "research" else "📰"
        
        md += f"{idx}. {category_emoji} **[{title}]({url})**\n"
        md += f"   - 날짜: {date} | 관련성: {relevance:.0%}\n"
        md += f"   - {article.get('summary', '')[:150]}...\n\n"
    
    if len(articles) > 10:
        md += f"_... 외 {len(articles)-10}개 자료 분석됨_\n\n"
    
    md += "---\n\n"
    
    # 3. 기술 트렌드 요약
    md += """## 🔬 기술 트렌드 요약

### 전체 요약

"""
    
    md += tech_summary.get("overall_summary", "다양한 기술 트렌드가 진행 중입니다.") + "\n\n"
    
    md += f"### 주요 기술 카테고리 ({len(tech_categories)}개)\n\n"
    
    for idx, cat in enumerate(tech_categories, 1):
        category_name = cat.get("category", "N/A")
        key_techs = cat.get("key_technologies", [])
        summary = cat.get("summary", "")
        applications = cat.get("applications", [])
        maturity = cat.get("maturity_level", "N/A")
        trend = cat.get("trend_direction", "N/A")
        
        md += f"#### {idx}. {category_name}\n\n"
        md += f"**주요 기술**: {_fmt_list(key_techs)}\n\n"
        md += f"**요약**: {summary}\n\n"
        md += f"**적용 분야**: {_fmt_list(applications)}\n\n"
        md += f"**성숙도**: {maturity} | **트렌드**: {trend}\n\n"
    
    md += "---\n\n"
    
    # 4. 트렌드 예측
    md += """## 🔮 트렌드 예측

### 전망 요약

"""
    
    md += predictions.get("overall_outlook", "기술 발전이 지속될 것으로 예상됩니다.") + "\n\n"
    
    md += f"### 기술별 예측 ({len(pred_list)}개)\n\n"
    
    for idx, pred in enumerate(pred_list, 1):
        category = pred.get("category", "N/A")
        market_impact = pred.get("market_impact", "N/A")
        adoption = pred.get("adoption_curve", "N/A")
        drivers = pred.get("key_drivers", [])
        barriers = pred.get("barriers", [])
        
        md += f"#### {idx}. {category}\n\n"
        md += f"**시장 영향도**: {market_impact} | **기술 채택 단계**: {adoption}\n\n"
        
        short = pred.get("short_term", {})
        mid = pred.get("mid_term", {})
        long = pred.get("long_term", {})
        
        md += "##### 🔹 단기 예측 (1년)\n\n"
        md += f"**신뢰도**: {short.get('confidence', 0):.0%}\n\n"
        md += f"{short.get('prediction', 'N/A')}\n\n"
        
        md += "##### 🔸 중기 예측 (3년)\n\n"
        md += f"**신뢰도**: {mid.get('confidence', 0):.0%}\n\n"
        md += f"{mid.get('prediction', 'N/A')}\n\n"
        
        md += "##### 🔺 장기 예측 (5년)\n\n"
        md += f"**신뢰도**: {long.get('confidence', 0):.0%}\n\n"
        md += f"{long.get('prediction', 'N/A')}\n\n"
        
        if drivers:
            md += f"**성장 동력**: {_fmt_list(drivers)}\n\n"
        if barriers:
            md += f"**장애 요인**: {_fmt_list(barriers)}\n\n"
    
    md += "---\n\n"
    
    # 5. 리스크 및 기회 분석
    md += """## ⚖️ 리스크 및 기회 분석

### 종합 요약

"""
    
    md += ro_analysis.get("summary", "리스크와 기회가 공존하는 상황입니다.") + "\n\n"
    
    md += f"### 기술별 분석 ({len(analyses)}개)\n\n"
    
    for idx, analysis in enumerate(analyses, 1):
        category = analysis.get("category", "N/A")
        opportunities = analysis.get("opportunities", [])
        risks = analysis.get("risks", [])
        swot = analysis.get("swot", {})
        
        md += f"#### {idx}. {category}\n\n"
        
        # 기회
        if opportunities:
            md += f"##### 💡 기회 요인 ({len(opportunities)}개)\n\n"
            for opp_idx, opp in enumerate(opportunities, 1):
                title = opp.get("title", "N/A")
                desc = opp.get("description", "")
                impact = opp.get("impact", "N/A")
                timeframe = opp.get("timeframe", "N/A")
                strategy = opp.get("exploitation_strategy", "N/A")
                
                md += f"**{opp_idx}. {title}**\n\n"
                md += f"- **설명**: {desc}\n"
                md += f"- **영향도**: {impact} | **시기**: {timeframe}\n"
                md += f"- **활용 전략**: {strategy}\n\n"
        
        # 리스크
        if risks:
            md += f"##### ⚠️ 리스크 요인 ({len(risks)}개)\n\n"
            for risk_idx, risk in enumerate(risks, 1):
                title = risk.get("title", "N/A")
                desc = risk.get("description", "")
                severity = risk.get("severity", "N/A")
                likelihood = risk.get("likelihood", "N/A")
                mitigation = risk.get("mitigation_strategy", "N/A")
                
                md += f"**{risk_idx}. {title}**\n\n"
                md += f"- **설명**: {desc}\n"
                md += f"- **심각도**: {severity} | **가능성**: {likelihood}\n"
                md += f"- **완화 전략**: {mitigation}\n\n"
        
        # SWOT
        if swot:
            md += "##### 📊 SWOT 분석\n\n"
            md += "| 강점 (Strengths) | 약점 (Weaknesses) |\n"
            md += "|---|---|\n"
            
            strengths = swot.get("strengths", [])
            weaknesses = swot.get("weaknesses", [])
            max_rows_sw = max(len(strengths), len(weaknesses))
            
            for i in range(max_rows_sw):
                s = strengths[i] if i < len(strengths) else ""
                w = weaknesses[i] if i < len(weaknesses) else ""
                md += f"| {s} | {w} |\n"
            
            md += "\n| 기회 (Opportunities) | 위협 (Threats) |\n"
            md += "|---|---|\n"
            
            opportunities_swot = swot.get("opportunities", [])
            threats = swot.get("threats", [])
            max_rows_ot = max(len(opportunities_swot), len(threats))
            
            for i in range(max_rows_ot):
                o = opportunities_swot[i] if i < len(opportunities_swot) else ""
                t = threats[i] if i < len(threats) else ""
                md += f"| {o} | {t} |\n"
            
            md += "\n"
    
    md += "---\n\n"
    
    # 6. 결론 및 제언
    md += """## 🎯 결론 및 제언

### 핵심 결론

"""
    
    # 결론 생성
    md += f"본 보고서는 {topic} 분야의 {len(articles)}개 최신 연구 및 뉴스를 분석하여 "
    md += f"{len(tech_categories)}개 주요 기술 트렌드를 식별하고, "
    md += "단기·중기·장기 전망과 함께 리스크 및 기회 요인을 체계적으로 분석하였습니다.\n\n"
    
    md += "### 주요 제언\n\n"
    
    # 제언 (기회 및 리스크 기반)
    recommendations = []
    
    # 기회 기반 제언
    for analysis in analyses[:3]:
        opps = analysis.get("opportunities", [])
        if opps and len(recommendations) < 3:
            opp = opps[0]
            if opp.get("impact") in ["높음", "매우높음"]:
                recommendations.append(
                    f"**{opp.get('title')}**: {opp.get('exploitation_strategy', '')}"
                )
    
    # 리스크 기반 제언
    for analysis in analyses[:3]:
        risks_list = analysis.get("risks", [])
        if risks_list and len(recommendations) < 5:
            risk = risks_list[0]
            if risk.get("severity") in ["높음", "매우높음"]:
                recommendations.append(
                    f"**{risk.get('title')} 대응**: {risk.get('mitigation_strategy', '')}"
                )
    
    if not recommendations:
        recommendations = [
            "지속적인 기술 모니터링 및 트렌드 추적",
            "주요 기술에 대한 선제적 투자 및 역량 확보",
            "리스크 관리 체계 구축 및 대응 전략 수립"
        ]
    
    for idx, rec in enumerate(recommendations, 1):
        md += f"{idx}. {rec}\n"
    
    md += "\n### 향후 모니터링 포인트\n\n"
    md += "- 주요 기술의 상용화 진척도 및 시장 반응\n"
    md += "- 규제 환경 변화 및 정책 동향\n"
    md += "- 경쟁 기술 출현 및 시장 구도 변화\n"
    md += "- 사용자 수용도 및 채택 속도\n\n"
    
    md += "---\n\n"
    
    # 7. 참고 자료
    md += """## 📚 참고 자료

### 데이터 출처

"""
    
    md += f"- **수집 기간**: {collection_date}\n"
    md += f"- **총 자료 수**: {len(articles)}개\n"
    md += f"- **출처**: {_fmt_list(sources, 20)}\n\n"
    
    md += "### 주요 참고 문헌\n\n"
    
    for idx, article in enumerate(articles[:20], 1):
        title = article.get("title", "N/A")
        url = article.get("url", "#")
        source = article.get("source", "N/A")
        date = article.get("published_date", "N/A")
        
        md += f"{idx}. [{title}]({url}) - {source} ({date})\n"
    
    if len(articles) > 20:
        md += f"\n_... 외 {len(articles)-20}개 참고 문헌_\n"
    
    md += "\n---\n\n"
    md += f"_보고서 작성: AI 트렌드 분석 에이전트 | {timestamp}_\n"
    
    return md


# ----------------------------- PDF Generator -----------------------------

class TrendReportPDFGenerator:
    """트렌드 분석 보고서 PDF 생성기"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not REPORTLAB_AVAILABLE:
            print("⚠️ reportlab이 설치되지 않아 PDF를 생성할 수 없습니다.")
            self.korean_font = None
            return
        
        # 한글 폰트 등록 (macOS AppleGothic)
        try:
            pdfmetrics.registerFont(TTFont('AppleGothic', '/System/Library/Fonts/Supplemental/AppleGothic.ttf'))
            self.korean_font = 'AppleGothic'
        except:
            print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
            self.korean_font = 'Helvetica'
    
    def create_styles(self):
        """PDF 스타일 정의"""
        if not REPORTLAB_AVAILABLE:
            return None
        
        styles = getSampleStyleSheet()
        
        # 제목
        styles.add(ParagraphStyle(
            name='KoreanTitle',
            fontName=self.korean_font,
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=30,
            textColor=colors.HexColor('#1a1a1a')
        ))
        
        # 부제목
        styles.add(ParagraphStyle(
            name='KoreanHeading',
            fontName=self.korean_font,
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#2c3e50'),
            bold=True
        ))
        
        # 소제목
        styles.add(ParagraphStyle(
            name='KoreanSubHeading',
            fontName=self.korean_font,
            fontSize=12,
            spaceBefore=10,
            spaceAfter=8,
            textColor=colors.HexColor('#34495e'),
            bold=True
        ))
        
        # 본문
        styles.add(ParagraphStyle(
            name='KoreanBodyText',
            fontName=self.korean_font,
            fontSize=10,
            alignment=TA_LEFT,
            leading=16,
            spaceAfter=6
        ))
        
        return styles
    
    def _clean_text(self, text):
        """텍스트 정리"""
        if not text:
            return ""
        import re
        text = re.sub(r'<[^>]+>', '', str(text))
        text = text.replace('<br/>', '\n').replace('<br>', '\n')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def generate_pdf(self, state: Dict[str, Any], filename: str) -> Optional[str]:
        """PDF 보고서 생성"""
        if not REPORTLAB_AVAILABLE or not self.korean_font:
            print("⚠️ PDF 생성을 건너뜁니다.")
            return None
        
        try:
            filepath = os.path.join(self.output_dir, filename.replace('.md', '.pdf'))
            
            # PDF 문서 생성
            doc = SimpleDocTemplate(
                filepath,
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )
            
            styles = self.create_styles()
            elements = []
            
            # 데이터 추출
            research_news = state.get("research_news", {})
            tech_summary = state.get("tech_trend_summary", {})
            predictions = state.get("trend_prediction", {})
            ro_analysis = state.get("risk_opportunity", {})
            
            topic = research_news.get("query", "기술 트렌드")
            
            # 제목
            title = Paragraph(f"🌐 {topic} 트렌드 분석 보고서", styles['KoreanTitle'])
            elements.append(title)
            elements.append(Spacer(1, 0.5*cm))
            
            # 날짜 정보 (시간까지 포함)
            datetime_full = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
            info_data = [
                ['분석 주제', topic],
                ['작성 일시', datetime_full],
                ['분석 자료', f"{len(research_news.get('articles', []))}개"],
            ]
            
            info_table = Table(info_data, colWidths=[5*cm, 12*cm])
            info_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), self.korean_font),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            
            elements.append(info_table)
            elements.append(Spacer(1, 0.8*cm))
            
            # I. 요약
            elements.append(Paragraph("I. 요약", styles['KoreanHeading']))
            elements.append(Spacer(1, 0.3*cm))
            
            key_insights = tech_summary.get("key_insights", [])
            if key_insights:
                insights_text = '<br/>'.join([f"• {insight}" for insight in key_insights[:5]])
                elements.append(Paragraph(insights_text, styles['KoreanBodyText']))
            
            elements.append(Spacer(1, 0.5*cm))
            
            outlook = predictions.get("overall_outlook", "긍정적 전망")
            elements.append(Paragraph(f"<b>전체 전망:</b> {self._clean_text(outlook)}", styles['KoreanBodyText']))
            elements.append(Spacer(1, 0.8*cm))
            
            # II. 기술 트렌드 요약
            elements.append(Paragraph("II. 기술 트렌드 요약", styles['KoreanHeading']))
            elements.append(Spacer(1, 0.3*cm))
            
            categories = tech_summary.get("technology_categories", [])
            for idx, cat in enumerate(categories[:5], 1):
                elements.append(Paragraph(f"{idx}. {cat.get('category', 'N/A')}", styles['KoreanSubHeading']))
                
                tech_text = f"<b>주요 기술:</b> {_fmt_list(cat.get('key_technologies', []))}<br/>"
                tech_text += f"<b>요약:</b> {self._clean_text(cat.get('summary', ''))}<br/>"
                tech_text += f"<b>성숙도:</b> {cat.get('maturity_level', 'N/A')} | "
                tech_text += f"<b>트렌드:</b> {cat.get('trend_direction', 'N/A')}"
                
                elements.append(Paragraph(tech_text, styles['KoreanBodyText']))
                elements.append(Spacer(1, 0.3*cm))
            
            elements.append(Spacer(1, 0.5*cm))
            
            # III. 트렌드 예측
            elements.append(Paragraph("III. 트렌드 예측", styles['KoreanHeading']))
            elements.append(Spacer(1, 0.3*cm))
            
            pred_list = predictions.get("predictions", [])
            for idx, pred in enumerate(pred_list[:3], 1):
                elements.append(Paragraph(f"{idx}. {pred.get('category', 'N/A')}", styles['KoreanSubHeading']))
                
                short = pred.get("short_term", {})
                mid = pred.get("mid_term", {})
                long = pred.get("long_term", {})
                
                pred_text = f"<b>단기 ({short.get('timeframe', '1년')}):</b> {self._clean_text(short.get('prediction', '')[:80])}<br/>"
                pred_text += f"<b>중기 ({mid.get('timeframe', '3년')}):</b> {self._clean_text(mid.get('prediction', '')[:80])}<br/>"
                pred_text += f"<b>장기 ({long.get('timeframe', '5년')}):</b> {self._clean_text(long.get('prediction', '')[:80])}"
                
                elements.append(Paragraph(pred_text, styles['KoreanBodyText']))
                elements.append(Spacer(1, 0.3*cm))
            
            elements.append(Spacer(1, 0.5*cm))
            
            # IV. 리스크 및 기회 분석
            elements.append(Paragraph("IV. 리스크 및 기회 분석", styles['KoreanHeading']))
            elements.append(Spacer(1, 0.3*cm))
            
            analyses = ro_analysis.get("analyses", [])
            for idx, analysis in enumerate(analyses[:3], 1):
                elements.append(Paragraph(f"{idx}. {analysis.get('category', 'N/A')}", styles['KoreanSubHeading']))
                
                opportunities = analysis.get("opportunities", [])
                risks = analysis.get("risks", [])
                
                if opportunities:
                    opp_text = "<b>주요 기회:</b><br/>"
                    for opp in opportunities[:2]:
                        opp_text += f"• {self._clean_text(opp.get('title', ''))}: {self._clean_text(opp.get('description', '')[:60])}<br/>"
                    elements.append(Paragraph(opp_text, styles['KoreanBodyText']))
                
                if risks:
                    risk_text = "<b>주요 리스크:</b><br/>"
                    for risk in risks[:2]:
                        risk_text += f"• {self._clean_text(risk.get('title', ''))}: {self._clean_text(risk.get('description', '')[:60])}<br/>"
                    elements.append(Paragraph(risk_text, styles['KoreanBodyText']))
                
                elements.append(Spacer(1, 0.3*cm))
            
            elements.append(Spacer(1, 0.5*cm))
            
            # V. 결론
            elements.append(Paragraph("V. 결론", styles['KoreanHeading']))
            elements.append(Spacer(1, 0.3*cm))
            
            summary_text = ro_analysis.get("summary", "기술 트렌드가 전반적으로 긍정적입니다.")
            elements.append(Paragraph(self._clean_text(summary_text), styles['KoreanBodyText']))
            
            # PDF 빌드
            doc.build(elements)
            
            print(f"✅ PDF 보고서가 생성되었습니다: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ PDF 생성 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None


# ----------------------------- Node Factory -----------------------------

def _trend_report_node_factory(cfg: TrendReportConfig):
    """LangGraph 노드 함수 팩토리"""
    
    def node(state: TrendReportState) -> TrendReportState:
        print(f"\n📄 트렌드 보고서 작성 시작")
        
        # 타임스탬프 (시간+분+초 포함으로 매번 고유한 파일명 생성)
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        datetime_str = now.strftime("%Y%m%d_%H%M%S")  # 날짜 + 시간 (예: 20251022_143052)
        
        # Markdown 보고서 생성
        markdown = _generate_markdown_report(state, timestamp)
        
        # 파일명 (실행할 때마다 고유한 파일명)
        research_news = state.get("research_news", {})
        topic = research_news.get("query", "tech_trend")
        topic_clean = topic.replace(" ", "_").replace("/", "_")[:30]
        filename = f"trend_report_{topic_clean}_{datetime_str}.md"
        
        # PDF 생성
        pdf_path = ""
        if cfg.generate_pdf:
            print(f"   📑 PDF 보고서 생성 중...")
            pdf_generator = TrendReportPDFGenerator(output_dir=cfg.output_dir)
            pdf_path = pdf_generator.generate_pdf(state, filename)
            if pdf_path:
                print(f"   ✅ PDF 저장 완료: {pdf_path}")
        
        state["trend_report"] = {
            "title": f"{research_news.get('query', '기술')} 트렌드 분석 보고서",
            "markdown": markdown,
            "filename": filename,
            "pdf_path": pdf_path or ""
        }
        
        # 결과 출력 (간소화)
        sections = markdown.count("##")
        
        print(f"\n   ✅ 보고서 생성 완료")
        print(f"   📝 파일: {filename}")
        if pdf_path:
            print(f"   📑 PDF: {os.path.basename(pdf_path)}")
        print(f"   📊 길이: {len(markdown):,}자 | 섹션: {sections}개")
        
        return state
    
    return node


# ----------------------------- Graph Builder -----------------------------

def build_trend_report_graph(config: Optional[TrendReportConfig] = None):
    cfg = config or TrendReportConfig()
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph가 설치되어 있지 않습니다. `pip install langgraph` 후 다시 시도하세요.")
    
    g = StateGraph(TrendReportState)
    g.add_node("trend_report_generator", _trend_report_node_factory(cfg))
    g.add_edge(START, "trend_report_generator")
    g.add_edge("trend_report_generator", END)
    return g.compile()


# ----------------------------- Helper -----------------------------

def run_trend_report_generator(state: Dict[str, Any], config: Optional[TrendReportConfig] = None) -> Dict[str, Any]:
    """트렌드 보고서 생성 실행"""
    app = build_trend_report_graph(config)
    return app.invoke(state)


def save_report_markdown(report_text: str, output_dir: str = "output", filename: str = "trend_report.md") -> str:
    """보고서를 Markdown 파일로 저장"""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"✅ 마크다운 보고서가 저장되었습니다: {file_path}")
    return file_path


# ----------------------------- Output Formatting -----------------------------

def print_trend_report_summary(result: Dict[str, Any]):
    """보고서 요약 정보 출력"""
    report = result.get("trend_report", {})
    
    print("\n" + "=" * 80)
    print("📄 트렌드 보고서 생성 완료")
    print("=" * 80)
    
    print(f"\n📋 제목: {report.get('title', 'N/A')}")
    print(f"📝 파일명: {report.get('filename', 'N/A')}")
    print(f"📊 길이: {len(report.get('markdown', '')):,} 자")
    
    if report.get("pdf_path"):
        print(f"📑 PDF: {report['pdf_path']}")
    
    print("\n" + "=" * 80)


# ----------------------------- CLI Test -----------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("📄 트렌드 보고서 생성 에이전트 테스트")
    print("=" * 80)
    
    # 모의 데이터
    dummy_state = {
        "research_news": {
            "query": "AI 기술",
            "collection_date": "2025-01-20",
            "articles": [
                {"title": "GPT-4 출시", "url": "https://example.com/1", "category": "news", 
                 "published_date": "2024", "summary": "GPT-4가 출시되었습니다.", "relevance_score": 0.9, "source": "example.com"}
            ],
            "sources": ["example.com"]
        },
        "tech_trend_summary": {
            "technology_categories": [
                {
                    "category": "생성형 AI",
                    "key_technologies": ["GPT-4"],
                    "summary": "빠른 발전",
                    "applications": ["콘텐츠 생성"],
                    "maturity_level": "성숙기",
                    "trend_direction": "상승"
                }
            ],
            "overall_summary": "AI 기술이 빠르게 발전하고 있습니다.",
            "key_insights": ["생성형 AI 확산", "다양한 산업 적용"]
        },
        "trend_prediction": {
            "predictions": [
                {
                    "category": "생성형 AI",
                    "short_term": {"timeframe": "1년", "prediction": "도입 가속화", "confidence": 0.8},
                    "mid_term": {"timeframe": "3년", "prediction": "산업 표준화", "confidence": 0.6},
                    "long_term": {"timeframe": "5년", "prediction": "전면 확산", "confidence": 0.5},
                    "market_impact": "매우높음",
                    "adoption_curve": "초기 대중화",
                    "key_drivers": ["기술 발전"],
                    "barriers": ["규제"]
                }
            ],
            "overall_outlook": "긍정적 전망"
        },
        "risk_opportunity": {
            "analyses": [
                {
                    "category": "생성형 AI",
                    "opportunities": [
                        {"title": "시장 성장", "description": "수요 증가", "impact": "높음", 
                         "timeframe": "단기", "exploitation_strategy": "투자 확대"}
                    ],
                    "risks": [
                        {"title": "규제 리스크", "description": "규제 강화", "severity": "보통", 
                         "likelihood": "높음", "mitigation_strategy": "컴플라이언스 강화"}
                    ],
                    "swot": {
                        "strengths": ["기술 우위"],
                        "weaknesses": ["높은 비용"],
                        "opportunities": ["시장 확대"],
                        "threats": ["경쟁 심화"]
                    }
                }
            ],
            "summary": "기회와 리스크 공존"
        }
    }
    
    try:
        config = TrendReportConfig(output_dir="output")
        final = run_trend_report_generator(dummy_state, config)
        print_trend_report_summary(final)
        
        # Markdown 저장
        report = final.get("trend_report", {})
        saved_path = save_report_markdown(
            report.get("markdown", ""),
            config.output_dir,
            report.get("filename", "trend_report.md")
        )
        
        print(f"\n💾 보고서가 저장되었습니다: {saved_path}")
        
        # 미리보기
        print("\n📝 보고서 미리보기 (처음 500자):")
        print("-" * 80)
        print(report.get("markdown", "")[:500])
        print("\n...")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

