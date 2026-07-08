"""
Maritime GraphRAG API — Q&A over the KMST marine-casualty knowledge graph.

The corpus is 139 real adjudication reports (재결서) of the Korean Maritime
Safety Tribunal, extracted into a two-layer graph:

  document layer   (Accident)-[:HAS_CHUNK]->(AContent {chunk, embedding})
  knowledge layer  (Accident)-[:INVOLVES]->(AVessel)  (Accident)-[:OCCURRED_IN]->(ALocation)
                   (Accident)-[:HAS_CAUSE]->(Cause)-[:OF_TYPE]->(CauseCategory)
                   (Cause)-[:LEADS_TO]->(Cause)   (Accident)-[:IMPOSED]->(Sanction)
                   (Accident)-[:CITES]->(Law)     (Accident)-[:ADJUDICATED_BY]->(Court)

Three retrievers sit behind an LLM router (ToolsRetriever):
  vector        semantic search over verdict text chunks
  vectorcypher  chunk hits expanded with the accident's causes/vessels/sanctions
  text2cypher   direct graph queries for aggregation and multi-hop questions

Every /search response also returns the evidence subgraph (nodes/edges) so the
frontend can show HOW the answer is connected: chunk -> accident -> causes.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import re
import neo4j
from dotenv import load_dotenv
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import (
    VectorRetriever, VectorCypherRetriever, Text2CypherRetriever, ToolsRetriever)
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation import RagTemplate, GraphRAG

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
rag = None
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

class QueryResponse(BaseModel):
    sections: List[Section]
    sources: List[Source]
    graph: Optional[SubGraph] = None


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


def initialize_graphrag():
    global driver, rag
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

    vector_retriever = VectorRetriever(
        driver=driver, index_name=INDEX_NAME, embedder=embedder)
    vector_cypher_retriever = VectorCypherRetriever(
        driver=driver, index_name=INDEX_NAME,
        retrieval_query=VECTOR_CYPHER_QUERY, embedder=embedder)
    text2cypher_retriever = Text2CypherRetriever(
        driver=driver, llm=llm, neo4j_schema=NEO4J_SCHEMA, examples=T2C_EXAMPLES)

    vector_tool = vector_retriever.convert_to_tool(
        name="vector_retriever",
        description="재결서 본문을 의미 기반으로 검색합니다. 특정 사고의 경위·상황 설명을 찾을 때 사용합니다.")
    vector_cypher_tool = vector_cypher_retriever.convert_to_tool(
        name="vectorcypher_retriever",
        description="재결서 본문 검색 결과에 해당 사고의 원인 사슬, 관련 선박, 장소, 처분을 함께 붙여 반환합니다. 사고의 전체 맥락이 필요할 때 사용합니다.")
    text2cypher_tool = text2cypher_retriever.convert_to_tool(
        name="text2cypher_retriever",
        description="자연어를 Cypher로 변환해 사고 그래프를 직접 질의합니다. 여러 사고에 걸친 집계(가장 흔한 원인, 건수, 평균 처분)나 조건 검색(야간·어선·특정 원인)에 사용합니다.")

    tools_retriever = ToolsRetriever(
        driver=driver, llm=llm,
        tools=[vector_tool, vector_cypher_tool, text2cypher_tool])

    prompt_template = RagTemplate(
        template="""당신은 해양안전심판원 재결서 139건의 지식그래프를 근거로 답하는 해양사고 분석 어시스턴트입니다.

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

답변:""",
        expected_inputs=["context", "query_text"])

    rag = GraphRAG(llm=llm, retriever=tools_retriever, prompt_template=prompt_template)
    load_entity_index(driver)
    print("✓ Maritime GraphRAG 초기화 완료 (재결서 코퍼스)")
    return True


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
async def search(request: QueryRequest):
    if not rag:
        raise HTTPException(status_code=503, detail="GraphRAG system not initialized")
    try:
        result = rag.search(query_text=request.query, return_context=True)

        answer_text = result.answer.strip()
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
                                    "content": result.answer, "sources": []}]}

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
            ctx_text = " ".join(str(i.content) for i in result.retriever_result.items) \
                if result.retriever_result else ""
            graph = build_subgraph(ctx_text + " " + result.answer + " " + request.query)
        except Exception as ge:
            print(f"서브그래프 생성 실패: {ge}")
            graph = None

        return {"sections": parsed.get("sections", []),
                "sources": sources, "graph": graph}
    except Exception as e:
        print(f"검색 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
