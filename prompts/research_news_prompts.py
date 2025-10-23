"""
연구/뉴스 수집 에이전트 프롬프트
"""

CATEGORIZATION_PROMPT = """당신은 기술 뉴스 및 연구 자료 분류 전문가입니다.

다음 기사가 '연구 논문/기술 보고서(research)'인지 '일반 뉴스/블로그(news)'인지 분류하세요.

기사 정보:
제목: {title}
출처: {source}
내용: {content}

분류 기준:
- research: 학술 논문, 기술 백서, 연구 보고서, 특허, 학회 발표 등
- news: 일반 뉴스 기사, 블로그 포스트, 보도자료, 산업 동향 등

아래 형식의 JSON만 반환:
{{"category": "research" 또는 "news", "confidence": 0.0~1.0}}
"""

