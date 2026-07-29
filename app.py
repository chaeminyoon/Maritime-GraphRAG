"""
Maritime GraphRAG API — Q&A over the KMST marine-casualty knowledge graph.

The corpus is 139 real adjudication reports (재결서) of the Korean Maritime
Safety Tribunal, extracted into a two-layer graph:

  document layer   (Accident)-[:HAS_CHUNK]->(AContent {chunk, embedding})
  knowledge layer  (Accident)-[:INVOLVES]->(AVessel)  (Accident)-[:OCCURRED_IN]->(ALocation)
                   (Accident)-[:HAS_CAUSE]->(Cause)-[:OF_TYPE]->(CauseCategory)
                   (Cause)-[:LEADS_TO]->(Cause)   (Accident)-[:IMPOSED]->(Sanction)
                   (Accident)-[:CITES]->(Law)     (Accident)-[:ADJUDICATED_BY]->(Court)

Three retrievers run as a best-of ensemble:
  vector        semantic search over verdict text chunks
  vectorcypher  chunk hits expanded with the accident's causes/vessels/sanctions
  text2cypher   direct graph queries for aggregation and multi-hop questions

All three run in parallel for every query; an LLM judge scores each context
for evidential support and the answer is generated from the winner. This
replaces upfront routing (predicting which retriever will be best) with
selection after the fact — a routing mistake can no longer send a query to
a retriever whose context cannot answer it.

Every /search response also returns the evidence subgraph (nodes/edges) so the
frontend can show HOW the answer is connected: chunk -> accident -> causes.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import neo4j
from dotenv import load_dotenv
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import (
    VectorRetriever, VectorCypherRetriever, Text2CypherRetriever)
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

load_dotenv()

URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "12345678"))
INDEX_NAME = "accident_chunk_index"
KMST_URL = "https://www.kmst.go.kr/web/verdictList.do?menuIdx=121"

app = FastAPI(title="Maritime GraphRAG API — KMST casualty knowledge graph")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

driver = None
llm = None
RETRIEVERS: Dict[str, object] = {}   # name -> retriever (ensemble members)
ENTITY_INDEX = []   # (name, label) pairs for matching answer text to graph nodes


# ---------------- response models ----------------
class QueryRequest(BaseModel):
    query: str

class Source(BaseModel):
    id: int
    shortName: str
    title: str
    category: str
    date: str
    url: str
    summary: str
    icon: str

class Section(BaseModel):
    title: str
    content: str
    sourceIds: List[int]

class GraphNode(BaseModel):
    id: str
    label: str
    type: str

class GraphEdge(BaseModel):
    source: str
    target: str
    type: str

class SubGraph(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]

class RetrievalInfo(BaseModel):
    method: str                      # winning retriever
    reason: str                      # judge's one-line justification
    scores: Optional[Dict[str, int]] = None   # judge scores per retriever
    errors: Optional[Dict[str, str]] = None   # retrievers that failed, if any

class QueryResponse(BaseModel):
    sections: List[Section]
    sources: List[Source]
    graph: Optional[SubGraph] = None
    retrieval: Optional[RetrievalInfo] = None


# ---------------- evidence subgraph ----------------
def load_entity_index(drv):
    global ENTITY_INDEX
    q = """
    MATCH (n) WHERE any(l IN labels(n) WHERE l IN
        ['AVessel','ALocation','CauseCategory','Court','Law','Accident'])
    RETURN coalesce(n.name, n.verdict_no) AS name, labels(n)[0] AS label
    """
    with drv.session() as s:
        ENTITY_INDEX = [(r['name'], r['label']) for r in s.run(q)
                        if r['name'] and len(r['name']) >= 2]
    print(f"엔티티 인덱스: {len(ENTITY_INDEX)}개")


def build_subgraph(texts: str, max_nodes: int = 40):
    """Evidence subgraph: retrieved chunks -> accidents -> causes/vessels/places."""
    nodes, edges, seen = [], [], set()

    def add_node(nid, label, ntype):
        if nid not in seen and len(nodes) < max_nodes:
            seen.add(nid)
            nodes.append({'id': nid, 'label': label, 'type': ntype})

    def add_edge(a, b, etype):
        if a in seen and b in seen:
            key = (a, b, etype)
            if key not in {(e['source'], e['target'], e['type']) for e in edges}:
                edges.append({'source': a, 'target': b, 'type': etype})

    chunk_ids = list(dict.fromkeys(re.findall(r"[A-Z]{2}\d{4}-?\d*#\d+", texts)))[:8]
    matched = [(n, l) for n, l in ENTITY_INDEX if n in texts]
    vnos = list(dict.fromkeys(
        [n for n, l in matched if l == 'Accident'] +
        [re.sub(r'#\d+$', '', c) for c in chunk_ids] +
        re.findall(r"[A-Z]{2}\d{4}-\d{3}(?!#)", texts)))[:12]
    cats = [n for n, l in matched if l == 'CauseCategory']
    vessels = [n for n, l in matched if l == 'AVessel']

    with driver.session() as s:
        # retrieved chunks (vector entry points)
        if chunk_ids:
            rows = s.run("""
                MATCH (c:AContent) WHERE c.content_id IN $cids
                MATCH (a:Accident)-[:HAS_CHUNK]->(c)
                RETURN c.content_id AS cid
            """, cids=chunk_ids).data()
            for r in rows:
                add_node('chunk:' + r['cid'], r['cid'].split('#')[0] + ' 본문', 'Chunk')

        # accidents matched by verdict_no / name / cause category / vessel
        rows = s.run("""
            MATCH (acc:Accident)
            WHERE acc.verdict_no IN $vnos OR acc.name IN $vnos
               OR EXISTS { MATCH (acc)-[:HAS_CAUSE]->(:Cause)-[:OF_TYPE]->(cat:CauseCategory)
                           WHERE cat.name IN $cats }
               OR EXISTS { MATCH (acc)-[:INVOLVES]->(v:AVessel) WHERE v.name IN $vessels }
            WITH acc, CASE WHEN acc.verdict_no IN $vnos OR acc.name IN $vnos
                           THEN 0 ELSE 1 END AS prio
            ORDER BY prio LIMIT 10
            OPTIONAL MATCH (acc)-[:HAS_CAUSE]->(:Cause)-[:OF_TYPE]->(cat:CauseCategory)
            OPTIONAL MATCH (acc)-[:INVOLVES]->(v:AVessel)
            OPTIONAL MATCH (acc)-[:OCCURRED_IN]->(loc:ALocation)
            RETURN acc.verdict_no AS vno, acc.name AS aname, acc.type AS atype,
                   collect(DISTINCT cat.name) AS cats,
                   collect(DISTINCT v.name) AS vessels,
                   collect(DISTINCT loc.name) AS locs
        """, vnos=vnos, cats=cats, vessels=vessels).data()
        for r in rows:
            acc_id = 'acc:' + r['vno']
            add_node(acc_id, (r['aname'] or r['vno'])[:24], 'Accident')
            for c in r['cats']:
                if c:
                    add_node('cat:' + c, c, 'CauseCategory')
                    add_edge(acc_id, 'cat:' + c, 'HAS_CAUSE')
            for v in r['vessels'][:3]:
                if v:
                    add_node('ves:' + v, v, 'AVessel')
                    add_edge(acc_id, 'ves:' + v, 'INVOLVES')
            for lo in r['locs'][:1]:
                if lo:
                    add_node('loc:' + lo, lo[:16], 'ALocation')
                    add_edge(acc_id, 'loc:' + lo, 'OCCURRED_IN')
            for cid in chunk_ids:
                if cid.startswith(r['vno']):
                    add_edge(acc_id, 'chunk:' + cid, 'HAS_CHUNK')

    if not edges:
        return None
    used = {e['source'] for e in edges} | {e['target'] for e in edges}
    nodes = [n for n in nodes if n['id'] in used]
    return {'nodes': nodes, 'edges': edges}


# ---------------- GraphRAG setup ----------------
NEO4J_SCHEMA = """
Nodes:
  Accident {verdict_no, name, type, date, night, weather, human_factors, keywords}
    # type: 충돌|접촉|좌초|전복|침몰|화재·폭발|기관손상|해양오염|인명사상|침수|운항저해|기타
    # night: true(야간)/false(주간)/null
  AContent {content_id, chunk}          # 재결서 본문 청크
  AVessel {name, type, gross_tonnage}   # type: 어선|화물선|유조선|여객선|예인선|부선|수상레저기구|기타
  ALocation {name}, Court {name}
  Cause {description, order}
  CauseCategory {name}   # 경계 소홀, 항행법규 위반, 조선 부적절, 정비·점검 소홀,
                         # 작업안전수칙 미준수, 안전관리체제 미흡, 기기취급 불량 등
  Sanction {type, months, target_role}, Law {name}
Relationships:
  (Accident)-[:HAS_CHUNK]->(AContent)
  (Accident)-[:INVOLVES]->(AVessel)
  (Accident)-[:OCCURRED_IN]->(ALocation)
  (Accident)-[:ADJUDICATED_BY]->(Court)
  (Accident)-[:HAS_CAUSE]->(Cause)
  (Cause)-[:OF_TYPE]->(CauseCategory)
  (Cause)-[:LEADS_TO]->(Cause)          # 판시된 원인 사슬 (선행 -> 직접 원인)
  (Accident)-[:IMPOSED]->(Sanction)
  (Accident)-[:CITES]->(Law)
Rules:
  - 선박 이름의 '호' 접미사는 name에 포함되지 않는다 ('삼우7호' -> AVessel name '삼우7')
  - RETURN 절에는 답과 함께 사고의 verdict_no, name도 반환하라
"""

T2C_EXAMPLES = [
    "USER INPUT: '충돌 사고에서 가장 흔한 원인 카테고리는?' "
    "QUERY: MATCH (a:Accident {type: '충돌'})-[:HAS_CAUSE]->(:Cause)-[:OF_TYPE]->(cat:CauseCategory) "
    "RETURN cat.name, count(*) AS n ORDER BY n DESC LIMIT 5",
    "USER INPUT: '경계 소홀이 원인인 사고와 선박을 알려줘' "
    "QUERY: MATCH (a:Accident)-[:HAS_CAUSE]->(:Cause)-[:OF_TYPE]->(:CauseCategory {name: '경계 소홀'}) "
    "MATCH (a)-[:INVOLVES]->(v:AVessel) RETURN a.verdict_no, a.name, v.name LIMIT 10",
    "USER INPUT: '야간에 발생한 충돌 사고는 몇 건인가?' "
    "QUERY: MATCH (a:Accident {type: '충돌', night: true}) RETURN count(*)",
    "USER INPUT: '어선이 관련된 사고의 원인 분포는?' "
    "QUERY: MATCH (a:Accident)-[:INVOLVES]->(:AVessel {type: '어선'}) "
    "MATCH (a)-[:HAS_CAUSE]->(:Cause)-[:OF_TYPE]->(cat:CauseCategory) "
    "RETURN cat.name, count(DISTINCT a) AS n ORDER BY n DESC",
    "USER INPUT: '업무정지 처분이 가장 무거웠던 사고는?' "
    "QUERY: MATCH (a:Accident)-[:IMPOSED]->(s:Sanction) WHERE s.months IS NOT NULL "
    "RETURN a.verdict_no, a.name, s.type, s.months ORDER BY s.months DESC LIMIT 5",
]

VECTOR_CYPHER_QUERY = """
WITH node AS content, score
MATCH (acc:Accident)-[:HAS_CHUNK]->(content)
OPTIONAL MATCH (acc)-[:HAS_CAUSE]->(cause:Cause)-[:OF_TYPE]->(cat:CauseCategory)
OPTIONAL MATCH (acc)-[:INVOLVES]->(v:AVessel)
OPTIONAL MATCH (acc)-[:OCCURRED_IN]->(loc:ALocation)
OPTIONAL MATCH (acc)-[:IMPOSED]->(s:Sanction)
RETURN content.content_id AS content_id,
       content.chunk AS chunk,
       acc.verdict_no AS verdict_no, acc.name AS accident_name,
       acc.type AS accident_type, acc.date AS date, acc.night AS night,
       collect(DISTINCT cat.name) AS cause_categories,
       collect(DISTINCT cause.description)[0..4] AS causes,
       collect(DISTINCT v.name + '(' + coalesce(v.type,'') + ')') AS vessels,
       collect(DISTINCT loc.name) AS locations,
       collect(DISTINCT coalesce(s.type,'') + coalesce(toString(s.months),''))[0..4] AS sanctions,
       score
"""


ANSWER_TEMPLATE = """당신은 해양안전심판원 재결서 139건의 지식그래프를 근거로 답하는 해양사고 분석 어시스턴트입니다.

질문: {query_text}

검색된 재결 정보:
{context}

지침:
1. 검색 결과에 근거해 질문에 직접 답하세요. 사고를 인용할 때는 재결번호와 사건명을 명시하세요.
2. 원인을 언급할 때는 원인 카테고리(경계 소홀, 작업안전수칙 미준수 등)를 사용하세요.
3. 검색 결과에 없는 내용은 절대 지어내지 마세요. 재결번호·사건명·날짜를 창작하는 것은 금지입니다.
4. 검색 결과가 비어 있거나 무관하면 content에 "관련 재결을 찾을 수 없습니다"라고 쓰고 sources는 빈 배열로 두세요.
5. 다음 JSON 형식으로만 답변하세요 (마크다운 코드 블록 없이):

{{
  "sections": [
    {{
      "title": "분석 결과",
      "content": "질문에 대한 직접 답변 (근거 사고의 재결번호 포함)",
      "sources": [
        {{
          "title": "사건명",
          "verdict_no": "재결번호",
          "date": "사고일 또는 재결일",
          "category": "사고유형",
          "court": "관할 심판원",
          "summary": "이 사고가 답의 근거가 되는 이유 (1-2문장)"
        }}
      ]
    }}
  ]
}}

답변:"""

# Judge context caps: enough to show what evidence a retriever found without
# paying for full chunks three times over.
JUDGE_CTX_CHARS = 3500
# Fallback when the judge cannot decide: highest measured accuracy on the
# retrieval benchmark (graph-aware retrieval), see evaluation/.
FALLBACK_METHOD = "vectorcypher"


def initialize_graphrag():
    global driver, llm
    print("Initializing Maritime GraphRAG (KMST accident graph)...")
    try:
        driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
        print("✓ Neo4j 연결 성공")
    except Exception as e:
        print(f"Neo4j 연결 실패: {e}")
        return False

    llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    RETRIEVERS["vector"] = VectorRetriever(
        driver=driver, index_name=INDEX_NAME, embedder=embedder)
    RETRIEVERS["vectorcypher"] = VectorCypherRetriever(
        driver=driver, index_name=INDEX_NAME,
        retrieval_query=VECTOR_CYPHER_QUERY, embedder=embedder)
    RETRIEVERS["text2cypher"] = Text2CypherRetriever(
        driver=driver, llm=llm, neo4j_schema=NEO4J_SCHEMA, examples=T2C_EXAMPLES)

    load_entity_index(driver)
    print("✓ Maritime GraphRAG 초기화 완료 (앙상블: vector / vectorcypher / text2cypher)")
    return True


# ---------------- best-of ensemble ----------------
def _retrieve(name: str, query: str):
    """Run one retriever, return (context_text, is_empty)."""
    retriever = RETRIEVERS[name]
    if name == "text2cypher":
        res = retriever.search(query_text=query)
        cypher = (res.metadata or {}).get("cypher", "")
        rows = "\n".join(str(i.content) for i in res.items if i.content)
        if not rows:
            return f"[실행한 그래프 질의] {cypher}\n[질의 결과] (0건)", True
        return f"[실행한 그래프 질의] {cypher}\n[질의 결과]\n{rows}", False
    res = retriever.search(query_text=query, top_k=5)
    text = "\n\n".join(str(i.content) for i in res.items if i.content)
    return text, not text


def run_all_retrievers(query: str) -> Dict[str, dict]:
    """Run every retriever in parallel; a failure disables that candidate
    instead of failing the request."""
    results = {}
    with ThreadPoolExecutor(max_workers=len(RETRIEVERS)) as pool:
        futures = {name: pool.submit(_retrieve, name, query) for name in RETRIEVERS}
        for name, future in futures.items():
            try:
                context, empty = future.result(timeout=90)
                results[name] = {"context": context, "empty": empty, "error": None}
            except Exception as e:
                print(f"retriever {name} 실패: {e}")
                results[name] = {"context": "", "empty": True, "error": str(e)}
    return results


def judge_select(query: str, results: Dict[str, dict]) -> dict:
    """Score each retriever's context for evidential support and pick the best.

    Returns {"method", "reason", "scores"}. Falls back to FALLBACK_METHOD
    (or the only live candidate) when the judge cannot run or answers
    something unusable.
    """
    candidates = [n for n, r in results.items() if not r["empty"]]
    if not candidates:
        return {"method": FALLBACK_METHOD, "reason": "모든 검색 결과가 비어 있음", "scores": None}
    if len(candidates) == 1:
        return {"method": candidates[0], "reason": "유일하게 결과를 반환한 검색 방법", "scores": None}

    blocks = []
    for name in candidates:
        ctx = results[name]["context"][:JUDGE_CTX_CHARS]
        blocks.append(f"### {name}\n{ctx}")
    prompt = (
        "당신은 검색 품질 심판입니다. 아래 질문에 답하기 위한 근거로 어느 검색 결과가 "
        "가장 충실한지 평가하세요.\n"
        "평가 기준:\n"
        "- 질문에 직접 답할 수 있는 정보(사고, 원인, 수치, 재결번호)가 실제로 들어 있는가\n"
        "- 집계·건수·순위 질문이면 그래프 질의 결과([질의 결과]에 행이 있는 경우)가 "
        "본문 발췌보다 신뢰할 수 있다\n"
        "- 질문과 무관하거나 근거가 빈약한 결과는 낮게 평가하라\n"
        f"각 결과에 0-10점을 매기고 최고점 하나를 고르세요. 후보: {', '.join(candidates)}\n"
        '다음 JSON만 출력하세요: {"scores": {"이름": 점수}, "best": "이름", "reason": "한 문장"}\n\n'
        f"[질문] {query}\n\n" + "\n\n".join(blocks)
    )
    try:
        raw = llm.invoke(prompt).content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        verdict = json.loads(raw)
        best = verdict.get("best")
        if best not in candidates:
            raise ValueError(f"judge picked unknown candidate: {best}")
        return {"method": best,
                "reason": verdict.get("reason", ""),
                "scores": {k: int(v) for k, v in (verdict.get("scores") or {}).items()
                           if k in results}}
    except Exception as e:
        print(f"심판 실패({e}) — 폴백: {FALLBACK_METHOD}")
        fallback = FALLBACK_METHOD if FALLBACK_METHOD in candidates else candidates[0]
        return {"method": fallback, "reason": f"심판 실패로 기본 검색 방법 사용 ({e})",
                "scores": None}


def ensemble_search(query: str):
    """Run all retrievers, judge the contexts, generate from the winner."""
    results = run_all_retrievers(query)
    choice = judge_select(query, results)
    context = results[choice["method"]]["context"]
    answer = llm.invoke(
        ANSWER_TEMPLATE.format(query_text=query, context=context or "(검색 결과 없음)")
    ).content
    errors = {n: r["error"] for n, r in results.items() if r["error"]}
    return answer, context, {**choice, "errors": errors or None}


@app.on_event("startup")
async def startup_event():
    if not initialize_graphrag():
        print("경고: GraphRAG 초기화 실패 — /search 사용 불가")


@app.get("/stats")
async def stats():
    """Corpus stats for the frontend meta strip."""
    if not driver:
        raise HTTPException(status_code=503, detail="not initialized")
    with driver.session() as s:
        row = s.run("""
            MATCH (a:Accident) WITH count(a) AS accidents
            MATCH (c:Cause) WITH accidents, count(c) AS causes
            MATCH (v:AVessel) WITH accidents, causes, count(v) AS vessels
            MATCH ()-[r:LEADS_TO]->() RETURN accidents, causes, vessels, count(r) AS chains
        """).single()
    return {"accidents": row["accidents"], "causes": row["causes"],
            "vessels": row["vessels"], "chains": row["chains"]}


@app.get("/health")
async def health():
    try:
        if driver:
            driver.verify_connectivity()
            return {"status": "healthy", "database": "connected"}
        return {"status": "unhealthy", "database": "not initialized"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/search", response_model=QueryResponse)
def search(request: QueryRequest):
    if not RETRIEVERS or llm is None:
        raise HTTPException(status_code=503, detail="GraphRAG system not initialized")
    try:
        answer, context_text, retrieval_info = ensemble_search(request.query)

        answer_text = answer.strip()
        if answer_text.startswith('```'):
            lines = answer_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            answer_text = '\n'.join(lines).strip()

        try:
            parsed = json.loads(answer_text)
        except Exception as e:
            print(f"JSON 파싱 실패: {e}")
            parsed = {"sections": [{"title": "분석 결과",
                                    "content": answer, "sources": []}]}

        sources, source_id = [], 1
        for section in parsed.get("sections", []):
            source_ids = []
            for src in section.get("sources", []):
                # drop placeholder sources the LLM sometimes emits for aggregates
                if not src.get("title") or src.get("verdict_no") in (None, "", "재결번호"):
                    continue
                sources.append({
                    "id": source_id,
                    "shortName": src.get("court", "해심"),
                    "title": src.get("title", ""),
                    "category": src.get("category", "기타"),
                    "date": src.get("date", "") or "",
                    "url": KMST_URL,
                    "summary": (f"[{src.get('verdict_no', '')}] " if src.get('verdict_no') else "")
                               + src.get("summary", ""),
                    "icon": "",
                })
                source_ids.append(source_id)
                source_id += 1
            section["sourceIds"] = source_ids
            section.pop("sources", None)

        try:
            graph = build_subgraph(context_text + " " + answer + " " + request.query)
        except Exception as ge:
            print(f"서브그래프 생성 실패: {ge}")
            graph = None

        return {"sections": parsed.get("sections", []),
                "sources": sources, "graph": graph,
                "retrieval": retrieval_info}
    except Exception as e:
        print(f"검색 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
