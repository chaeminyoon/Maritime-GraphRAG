<p align="center">
  <a href="README.md">English</a> | <a href="README.ko.md">한국어</a>
</p>

<h1 align="center">Maritime GraphRAG</h1>

<p align="center">해양 도메인(선박·선사·항만·규제·사고) 특화 Neo4j 지식그래프 RAG 엔진 — 정답이 있는 멀티홉 검색 벤치마크 포함 (strict accuracy: 벡터 0.70 → 그래프 질의 0.85)</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Neo4j-Knowledge_Graph-008CC1?logo=neo4j&logoColor=white" alt="Neo4j">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB?logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/Benchmark-0.70%E2%86%920.85-green.svg" alt="Benchmark">
</p>

해양 산업의 질문은 본질적으로 관계형입니다: *"부산항에 컨테이너선을 기항시키는
선사는?"* 은 선사→선박→항만 조인이고, *"울산항 사고 선박에 적용되는 환경 규제는?"*
은 사고→선박→선종→규제 체인입니다. 순수 벡터 검색은 각 엔티티에 대한 문서는
찾아도 이 조인을 수행하지 못합니다. 이 프로젝트는 해양 뉴스 위에 2계층 Neo4j
그래프(문서 + 타입 엔티티)를 구축하고, 에이전틱 검색기 라우터(FastAPI + React)로
서빙하며, **그래프가 실제로 얼마나 기여하는지 측정**합니다.

## 검색 벤치마크 — 그래프는 값을 하는가?

실제 뉴스에는 정답이 없으므로, 저장소에 **합성 해양 세계**(가상 선사 6, 선박 14,
항만 6, 규제 4, 사고 6)를 동봉했습니다 — 모든 관계가 생성 시점에 확정되어
있습니다. 뉴스 스타일 기사 42건이 이 관계를 서술하고, 관계 테이블에서 QA 20문항을
도출했습니다. 멀티홉 문항의 답은 **어느 단일 기사에도 존재하지 않습니다**.
무검색(no-retrieval) 대조군으로 가상 엔티티가 GPT-4o의 사전지식으로는 답할 수
없음을 확인했습니다.

같은 LLM(gpt-4o, temp 0), 같은 답변 프롬프트에서 검색 방식만 바꿔 비교합니다.
채점 = 생성된 답변에 정답 엔티티가 포함됐는가.
`python evaluation/retrieval_benchmark.py` 로 재현:

| 검색 방식 | 엔티티 recall | Strict acc | 1홉 (n=11) | 2홉 (n=8) | 3홉 (n=1) |
|---|---|---|---|---|---|
| 무검색 (대조군) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 벡터 (top-5 청크) | 0.73 | 0.70 | 0.91 | 0.38 | 1.00 |
| 그래프 확장 (VectorCypher) | 0.82 | 0.75 | 0.91 | 0.50 | 1.00 |
| **Text2Cypher (그래프 직접 질의)** | **0.85** | **0.85** | **1.00** | **0.75** | 0.00 |

![Retrieval benchmark](docs/analysis/retrieval_benchmark.png)

**수치가 말하는 것:**

- **벡터 검색은 멀티홉에서 무너집니다** (1홉 0.91 → 2홉 0.38): 조인된 답 집합이
  검색 가능한 어떤 청크에도 없기 때문입니다.
- **그래프 확장이 일부를 회복합니다** (0.50): 검색된 청크에 언급 엔티티의 관계
  사실(`X -OPERATES- Y`)을 덧붙이면 LLM이 컨텍스트 안에서 일부 조인을 수행합니다.
- **그래프 직접 질의가 전체 최고입니다** (strict 0.85, 1홉 1.00) — 질문이 Cypher
  패턴에 대응되면 조인이 LLM이 아니라 데이터베이스에서 일어납니다.
- **단, Text2Cypher는 복잡한 체인에 취약합니다**: 3홉 문항과 일부 2홉 변형에서
  형태는 그럴듯하지만 결과가 빈 Cypher가 생성됐습니다(더 싼 검색기가 맞힌 문제에서
  0점). 어떤 검색기도 전 구간을 지배하지 못한다는 것 — 이것이 질문마다 전략을
  고르는 앱의 **에이전틱 라우터**(`ToolsRetriever`)의 실증적 근거입니다.

과정에서 발견·수정한 것: Text2Cypher 답변의 근거화(grounding)에는 값과 함께
**주체를 반환**해야 합니다(`RETURN v.name, v.type` — 값만 반환하면 답변 LLM이
근거 불충분으로 거부). 또 Neo4j 자동 추출 스키마는 잘못된 Cypher를 유발해,
`app.py`의 큐레이션된 스키마 텍스트가 벤치마크로 검증된 버전입니다.

## 그래프 스키마

```
문서 계층    (Article)-[:HAS_CHUNK]->(Content {embedding})
             (Article)-[:PUBLISHED_BY]->(Media)   (Article)-[:BELONGS_TO]->(Category)
             (Article)-[:MENTIONS]->(엔티티)
지식 계층    (Company)-[:OPERATES]->(Vessel {type})
             (Vessel)-[:CALLS_AT]->(Port)
             (Regulation)-[:APPLIES_TO]->(Vessel)
             (Vessel)-[:INVOLVED_IN]->(Incident)-[:OCCURRED_AT]->(Port)
```

동봉 코퍼스의 지식 계층은 정답 테이블(`data/corpus/entities.json`)에서 적재하고,
실제 문서에는 `ingest/extract_entities.py`(LLM 추출기)가 같은 구조를 생성합니다.

## 애플리케이션

FastAPI 백엔드가 3개 검색기를 LLM 라우터(`ToolsRetriever`) 뒤에 두고 질문마다
벡터 / 그래프 확장 / Text2Cypher를 선택하며, 인용 근거가 있는 JSON 계약으로
답해 React 프론트엔드가 렌더링합니다:

| 검색 화면 | 멀티홉 답변 |
|---|---|
| ![Search screen](docs/images/search_screen.png) | ![Result screen](docs/images/result_screen.png) |

## 빠른 시작

```bash
# 0. Neo4j
docker run -d --name maritime-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/maritime123 neo4j:5

# 1. 파이썬 환경 + 키
pip install -r requirements.txt
cp .env.example .env            # OPENAI_API_KEY, NEO4J_PASSWORD 설정

# 2. 합성 코퍼스 생성 + 그래프 구축 (청크 42개 임베딩)
python ingest/generate_corpus.py
python graph/build_graph.py

# 3. 벤치마크 실행
python evaluation/retrieval_benchmark.py

# 4. 서빙
uvicorn app:app --port 8001               # 백엔드
cd frontend && npm install && npm run dev # 프론트엔드 (localhost:5173)
```

## 저장소 구조

| 경로 | 내용 |
|---|---|
| `ingest/generate_corpus.py` | 합성 해양 코퍼스 + 관계 테이블 + QA 벤치마크 (결정론적) |
| `ingest/extract_entities.py` | 실제 문서용 LLM 엔티티/관계 추출기 (동일 출력 스키마) |
| `graph/build_graph.py` | 2계층 Neo4j 구축: 문서·청크·임베딩·타입 엔티티 |
| `evaluation/retrieval_benchmark.py` | 4개 검색 전략 비교, 정답 엔티티 채점 |
| `app.py` | FastAPI + 에이전틱 검색기 라우터 + 인용 근거 생성 |
| `frontend/` | React (Vite) 검색 UI |
| `data/` | 커밋된 코퍼스, 엔티티 테이블, QA 벤치마크 |

## 한계

- 벤치마크 코퍼스는 합성이며 소규모입니다(기사 42, QA 20). 수치는 동일 조건에서의
  검색 전략 간 비교이지 절대적 프로덕션 품질 추정이 아닙니다. 기사가 템플릿 문체라
  실제 뉴스의 패러프레이즈·노이즈가 더해지면 전 행이 하락하며, 벡터가 가장 크게
  하락할 가능성이 높습니다.
- 수치는 temperature 0의 단일 실행입니다. 특히 Text2Cypher 실패는 프롬프트 표현에
  따라 변동합니다.
- 데모의 지식 계층은 정답 테이블에서 적재되어 검색 품질을 추출 노이즈와 분리해
  측정합니다. 실데이터에 `extract_entities.py`를 쓰면 추출 오류가 이 수치 위에
  더해집니다.

## 관련 프로젝트

- [Parse-Everything](https://github.com/chaeminyoon/Parse-Everything) — 자가 치유 문서 파싱 (RAG 코퍼스의 상류 공정)
- [Vehicle-Anomaly-Algorithm](https://github.com/chaeminyoon/Vehicle-Anomaly-Algorithm) · [AIS-Traffic-Model](https://github.com/chaeminyoon/AIS-Traffic-Model) · [cbm-anomaly-detection](https://github.com/chaeminyoon/cbm-anomaly-detection) — 이 프로젝트가 지식 검색으로 확장하는 해양·교통 AI 라인
