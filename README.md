# AI 기술 트렌드 분석 에이전트 시스템

## Overview

- **Objective**: 최신 기술 트렌드를 자동으로 수집·분석하고 종합 보고서를 생성
- **Methods**: LangGraph 기반 순차적 에이전트 워크플로우, AI 구조화된 출력
- **Tools**: Tavily API, OpenAI GPT-4o-mini, Pydantic 스키마, ReportLab PDF

## Features

- **자동화된 연구/뉴스 수집**: Tavily API를 통한 고품질 자료 수집 및 출처별 분류
- **AI 기반 트렌드 분석**: GPT-4o-mini를 활용한 기술 트렌드 요약 및 예측
- **종합 보고서 생성**: Markdown 및 PDF 형식의 전문 보고서 자동 생성

## Tech Stack

| Category | Details |
| --- | --- |
| Framework | LangGraph, LangChain, Python |
| LLM | GPT-4o-mini via OpenAI API |
| Search | Tavily API for real-time web search |
| Validation | Pydantic for structured output |
| PDF | ReportLab with Korean font support |

## Agents

- **Research News Collector**: 최신 연구/뉴스 수집 및 출처별 분류
- **Tech Trend Summary**: 수집된 정보를 바탕으로 기술별 트렌드 요약
- **Trend Prediction**: 단기/중기/장기 트렌드 예측 및 시장 영향도 분석
- **Risk Opportunity Analysis**: 리스크 및 기회 요인 분석 (SWOT 포함)
- **Trend Report Generator**: 종합 보고서 생성 (Markdown/PDF)

## State

- **topic**: 분석할 기술 주제/키워드
- **time_range**: 분석 시간 범위 (예: "1-2 years", "4-5 years")
- **max_results**: 최대 수집 결과 수
- **search_depth**: 검색 깊이 (basic|advanced)
- **research_news**: 수집된 연구/뉴스 자료 (제목, URL, 출처, 요약, 관련성 점수)
- **tech_trends**: 기술별 트렌드 요약 (카테고리, 성숙도, 트렌드 방향)
- **trend_predictions**: 단기/중기/장기 예측 (신뢰도, 시장 영향도)
- **risk_opportunity**: 리스크/기회 분석 (SWOT, 대응 전략)
- **final_report**: 최종 보고서 (Markdown/PDF 파일 경로)

## 기술 스택

- **LangGraph**: 에이전트 워크플로우 orchestration
- **LangChain**: LLM 통합 및 체인 구성
- **OpenAI GPT-4o-mini**: 자연어 분석 및 생성
- **Tavily API**: 웹 검색 및 뉴스 수집
- **Pydantic**: 데이터 검증 및 스키마 정의

## 워크플로우

```
[입력] 주제, 시간범위, 검색 옵션
   ↓
[1단계] 연구/뉴스 수집 에이전트
   ├─ Tavily 웹 검색
   ├─ 관련성 필터링
   └─ 출처별 분류
   ↓
[2단계] 기술 트렌드 요약 에이전트
   ├─ 기술 카테고리 분류
   ├─ LLM 기반 요약
   └─ 성숙도 평가
   ↓
[3단계] 트렌드 예측 에이전트
   ├─ 단기/중기/장기 예측
   ├─ 신뢰도 계산
   └─ 시장 영향도 분석
   ↓
[4단계] 리스크/기회 분석 에이전트
   ├─ 기회 요인 식별
   └─ 리스크 평가
   ↓
[5단계] 보고서 생성 에이전트
   ├─ Markdown 보고서 작성
   └─ 파일 저장
   ↓
[출력] 종합 트렌드 분석 보고서
```
