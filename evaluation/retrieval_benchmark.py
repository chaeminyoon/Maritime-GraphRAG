"""
Retrieval benchmark: pure vector search vs graph-aware retrieval vs Text2Cypher,
plus the two orchestration strategies built on top of them.

Runs the ground-truth QA set (data/benchmark/qa_benchmark.json) against six
configurations and scores the GENERATED ANSWER against gold entities:

  no-retrieval   gpt-4o with no context (control: entities are fictional, so
                 this must score ~0 — proves the benchmark cannot be answered
                 from parametric knowledge)
  vector         VectorRetriever over Content chunks (top_k=5)
  graph          VectorCypherRetriever: vector entry point, then expand
                 Article -> MENTIONS -> entity -> typed relations (1 hop)
                 and return chunks + structured facts
  text2cypher    Text2CypherRetriever: LLM writes Cypher against the schema
  router         ToolsRetriever: an LLM predicts which single retriever to
                 run per question (upfront routing — the app's old strategy)
  ensemble       run all three retrievers in parallel, an LLM judge scores
                 each context for evidential support, generate from the
                 winner (best-of selection — the app's current strategy)

Metrics per configuration:
  entity recall  mean fraction of gold entities present in the answer
  strict acc     fraction of questions with ALL gold entities present
  by hops        the same, split by reasoning depth (1 / 2 / 3 hops)

Thresholds/prompts identical across configurations; the only variable is
retrieval. Requires a built graph (graph/build_graph.py) and OPENAI_API_KEY.

Usage: python evaluation/retrieval_benchmark.py
Saves docs/analysis/retrieval_benchmark.png and prints the metric table.
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import neo4j
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import (
    VectorRetriever, VectorCypherRetriever, Text2CypherRetriever, ToolsRetriever)
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI

load_dotenv()

URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "12345678"))
INDEX_NAME = "content_vector_index"
TOP_K = 5
BENCH = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmark', 'qa_benchmark.json')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)

client = OpenAI()

ANSWER_PROMPT = """당신은 해양 산업 정보 어시스턴트입니다.
아래 컨텍스트만 근거로 질문에 답하세요. 답에 해당하는 엔티티 이름(선사/선박/항만/규제 등)을 빠짐없이 명시하세요.
컨텍스트에 근거가 없으면 "정보 없음"이라고 답하세요.

[컨텍스트]
{context}

[질문] {question}
[답변]"""


def generate_answer(question, context):
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user",
                   "content": ANSWER_PROMPT.format(context=context or "(없음)",
                                                   question=question)}])
    return resp.choices[0].message.content


def score(answer, gold_entities):
    """Fraction of gold entities present in the answer (word-boundary for numbers)."""
    hits = 0
    for g in gold_entities:
        if g.isdigit():
            if re.search(rf'(?<!\d){g}(?!\d)', answer):
                hits += 1
        elif g in answer:
            hits += 1
    return hits / len(gold_entities)


# ---------------- retriever setups ----------------
driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

vector_retriever = VectorRetriever(driver=driver, index_name=INDEX_NAME, embedder=embedder)

GRAPH_EXPANSION = """
WITH node AS content, score
MATCH (content)<-[:HAS_CHUNK]-(article:Article)
OPTIONAL MATCH (article)-[:MENTIONS]->(e)
OPTIONAL MATCH (e)-[r:OPERATES|CALLS_AT|APPLIES_TO|INVOLVED_IN|OCCURRED_AT]-(nb)
WITH content, score, article,
     collect(DISTINCT CASE WHEN e IS NULL THEN NULL ELSE
         coalesce(e.name, e.incident_id) END) AS mentioned,
     collect(DISTINCT CASE WHEN nb IS NULL THEN NULL ELSE
         coalesce(e.name, e.incident_id) + ' -' + type(r) + '- ' +
         coalesce(nb.name, nb.incident_id) END) AS facts
RETURN content.chunk AS chunk, article.title AS title,
       [m IN mentioned WHERE m IS NOT NULL] AS mentioned,
       [f IN facts WHERE f IS NOT NULL] AS facts, score
"""
graph_retriever = VectorCypherRetriever(
    driver=driver, index_name=INDEX_NAME,
    retrieval_query=GRAPH_EXPANSION, embedder=embedder)

SCHEMA = """
Nodes:
  Article {article_id, title, url, published_date}
  Content {content_id, chunk}
  Media {name}, Category {name}
  Vessel {name, type}   # type: 컨테이너선|유조선|LNG운반선|벌크선
  Company {name}, Port {name}
  Regulation {name, applies_to_type, description}
  Incident {incident_id, description}
Naming rules:
  - 선박 이름 뒤의 '호'는 저장된 name에 포함되지 않는다 (질문의 '대양스피릿호' -> Vessel name '대양스피릿')
  - RETURN 절에는 답이 되는 값과 함께 그 주체의 이름(v.name 등)도 반환하라
Relationships:
  (Article)-[:HAS_CHUNK]->(Content)
  (Article)-[:PUBLISHED_BY]->(Media)
  (Article)-[:BELONGS_TO]->(Category)
  (Article)-[:MENTIONS]->(Vessel|Company|Port|Regulation|Incident)
  (Company)-[:OPERATES]->(Vessel)
  (Vessel)-[:CALLS_AT]->(Port)
  (Regulation)-[:APPLIES_TO]->(Vessel)
  (Vessel)-[:INVOLVED_IN]->(Incident)
  (Incident)-[:OCCURRED_AT]->(Port)
"""
T2C_EXAMPLES = [
    "USER INPUT: '한서파이오니어호를 운영하는 선사는?' "
    "QUERY: MATCH (c:Company)-[:OPERATES]->(v:Vessel {name: '한서파이오니어'}) RETURN v.name, c.name",
    "USER INPUT: '청림웨이브호는 어떤 종류의 선박인가?' "
    "QUERY: MATCH (v:Vessel {name: '청림웨이브'}) RETURN v.name, v.type",
    "USER INPUT: '부산항에 기항하는 컨테이너선을 운영하는 선사는?' "
    "QUERY: MATCH (c:Company)-[:OPERATES]->(v:Vessel {type: '컨테이너선'})-[:CALLS_AT]->(:Port {name: '부산항'}) RETURN DISTINCT c.name",
    "USER INPUT: '울산항에서 발생한 사고는 몇 건인가?' "
    "QUERY: MATCH (:Incident)-[:OCCURRED_AT]->(:Port {name: '울산항'}) RETURN count(*)",
    "USER INPUT: '사고 이력이 있는 선박에 적용되는 규제는?' "
    "QUERY: MATCH (r:Regulation)-[:APPLIES_TO]->(v:Vessel)-[:INVOLVED_IN]->(:Incident) RETURN DISTINCT r.name",
]
t2c_llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})
t2c_retriever = Text2CypherRetriever(
    driver=driver, llm=t2c_llm, neo4j_schema=SCHEMA, examples=T2C_EXAMPLES)


def ctx_vector(question):
    items = vector_retriever.search(query_text=question, top_k=TOP_K).items
    return "\n\n".join(i.content for i in items)


def ctx_graph(question):
    items = graph_retriever.search(query_text=question, top_k=TOP_K).items
    return "\n\n".join(i.content for i in items)


def ctx_t2c(question):
    res = t2c_retriever.search(query_text=question)
    cypher = (res.metadata or {}).get('cypher', '')
    rows = "\n".join(i.content for i in res.items)
    if not rows:
        return f"[실행한 그래프 질의] {cypher}\n[질의 결과] (0건)"
    return f"[실행한 그래프 질의] {cypher}\n[질의 결과]\n{rows}"


# -------- router (upfront prediction — the app's old strategy) --------
router_retriever = ToolsRetriever(
    driver=driver, llm=t2c_llm,
    tools=[
        vector_retriever.convert_to_tool(
            name="vector_retriever",
            description="기사 본문을 의미 기반으로 검색합니다. 특정 사건의 서술적 설명을 찾을 때 사용합니다."),
        graph_retriever.convert_to_tool(
            name="graph_retriever",
            description="본문 검색 결과에 언급된 엔티티와 그 관계(운영, 기항, 규제, 사고)를 함께 붙여 반환합니다. 연결된 사실이 필요할 때 사용합니다."),
        t2c_retriever.convert_to_tool(
            name="text2cypher_retriever",
            description="자연어를 Cypher로 변환해 그래프를 직접 질의합니다. 집계(건수, 순위)나 다중 홉 조건 검색에 사용합니다."),
    ])


def ctx_router(question):
    items = router_retriever.search(query_text=question).items
    return "\n\n".join(str(i.content) for i in items if i.content)


# -------- ensemble (run all + judge selection — the app's current strategy) --------
JUDGE_CTX_CHARS = 3500
ENSEMBLE_PICKS = []   # winning method per question, printed after the run


def judge_pick(question, contexts):
    """Score each context for evidential support; return the winning name."""
    candidates = {n: c for n, c in contexts.items() if c and c.strip()
                  and '[질의 결과] (0건)' not in c}
    if not candidates:
        return 'graph'
    if len(candidates) == 1:
        return next(iter(candidates))
    blocks = "\n\n".join(f"### {n}\n{c[:JUDGE_CTX_CHARS]}" for n, c in candidates.items())
    prompt = (
        "당신은 검색 품질 심판입니다. 아래 질문에 답하기 위한 근거로 어느 검색 결과가 "
        "가장 충실한지 평가하세요.\n"
        "- 질문에 직접 답할 수 있는 정보(엔티티 이름, 수치)가 실제로 들어 있는가\n"
        "- 집계·건수·순위 질문이면 그래프 질의 결과가 본문 발췌보다 신뢰할 수 있다\n"
        f"후보: {', '.join(candidates)}\n"
        '다음 JSON만 출력하세요: {"best": "이름", "reason": "한 문장"}\n\n'
        f"[질문] {question}\n\n{blocks}")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o", temperature=0,
            messages=[{"role": "user", "content": prompt}])
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.choices[0].message.content.strip())
        best = json.loads(raw).get('best')
        if best in candidates:
            return best
    except Exception as e:
        print(f"  (judge error: {e})")
    return 'graph' if 'graph' in candidates else next(iter(candidates))


def ctx_ensemble(question):
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {n: pool.submit(f, question)
                   for n, f in (('vector', ctx_vector), ('graph', ctx_graph),
                                ('text2cypher', ctx_t2c))}
        contexts = {}
        for n, fut in futures.items():
            try:
                contexts[n] = fut.result(timeout=90)
            except Exception as e:
                contexts[n] = ""
                print(f"  ({n} error: {e})")
    picked = judge_pick(question, contexts)
    ENSEMBLE_PICKS.append(picked)
    return contexts[picked]


CONFIGS = {
    'no-retrieval': lambda q: "",
    'vector': ctx_vector,
    'graph (VectorCypher)': ctx_graph,
    'text2cypher': ctx_t2c,
    'router': ctx_router,
    'ensemble (judge)': ctx_ensemble,
}

# ---------------- run ----------------
with open(BENCH, encoding='utf-8') as f:
    qa = json.load(f)

results = {name: [] for name in CONFIGS}
for name, get_ctx in CONFIGS.items():
    print(f"\n=== {name} ===")
    for item in qa:
        t0 = time.time()
        try:
            ctx = get_ctx(item['question'])
            answer = generate_answer(item['question'], ctx)
            s = score(answer, item['gold_entities'])
        except Exception as e:
            answer, s = f"(error: {e})", 0.0
        dt = time.time() - t0
        results[name].append({'hops': item['hops'], 'score': s, 'latency': dt,
                              'q': item['question'], 'answer': answer[:200]})
        print(f"  [{s:.2f}] ({item['hops']}-hop) {item['question'][:44]}")

# ---------------- metrics ----------------
def agg(rows):
    recall = float(np.mean([r['score'] for r in rows]))
    strict = float(np.mean([r['score'] == 1.0 for r in rows]))
    return recall, strict

print(f"\n{'config':<22} {'recall':>7} {'strict':>7}   " +
      " ".join(f"{h}-hop" for h in (1, 2, 3)))
table = {}
for name, rows in results.items():
    recall, strict = agg(rows)
    by_hops = {h: agg([r for r in rows if r['hops'] == h]) for h in (1, 2, 3)}
    lat = float(np.mean([r['latency'] for r in rows]))
    table[name] = dict(recall=recall, strict=strict, by_hops=by_hops, latency=lat)
    print(f"{name:<22} {recall:>7.2f} {strict:>7.2f}   " +
          " ".join(f"{by_hops[h][1]:.2f}" for h in (1, 2, 3)) +
          f"   (avg {lat:.1f}s)")

if ENSEMBLE_PICKS:
    from collections import Counter
    print(f"ensemble picks: {dict(Counter(ENSEMBLE_PICKS))}")

with open(os.path.join(OUT_DIR, 'benchmark_details.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# ---------------- figure ----------------
fig, axs = plt.subplots(1, 2, figsize=(16, 5))
names = list(CONFIGS)
colors = ['#94A3B8', '#2E6F9E', '#0F4C81', '#C4762E', '#7A5CA8', '#2F7D5C']

x = np.arange(len(names))
axs[0].bar(x - 0.18, [table[n]['recall'] for n in names], 0.36,
           color=colors, alpha=0.55, label='entity recall')
axs[0].bar(x + 0.18, [table[n]['strict'] for n in names], 0.36,
           color=colors, label='strict accuracy')
axs[0].set_xticks(x)
axs[0].set_xticklabels(names, rotation=12, fontsize=8)
axs[0].set_ylim(0, 1.05)
axs[0].set_title('Answer quality by retrieval strategy (20 QA, gold entities)')
axs[0].legend(fontsize=9)
axs[0].grid(alpha=0.3, axis='y')

hops = [1, 2, 3]
w = 0.13
for i, n in enumerate(names):
    axs[1].bar(np.arange(len(hops)) + (i - (len(names) - 1) / 2) * w,
               [table[n]['by_hops'][h][1] for h in hops], w,
               color=colors[i], label=n)
axs[1].set_xticks(np.arange(len(hops)))
axs[1].set_xticklabels([f'{h}-hop\n(n={sum(1 for q in qa if q["hops"] == h)})' for h in hops])
axs[1].set_ylim(0, 1.05)
axs[1].set_title('Strict accuracy by reasoning depth')
axs[1].legend(fontsize=8)
axs[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
out = os.path.join(OUT_DIR, 'retrieval_benchmark.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"\nsaved: {out}")
print("BENCHMARK_OK")
