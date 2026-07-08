"""
Load the extracted KMST accident records into Neo4j.

Schema (coexists with the synthetic-news graph; separate labels):

  (Accident {verdict_no, name, type, date, night, weather})
      -[:INVOLVES {role}]->        (AVessel {name, type, gross_tonnage})
      -[:OCCURRED_IN]->            (ALocation {name})
      -[:ADJUDICATED_BY]->         (Court {name})
      -[:HAS_CAUSE]->              (Cause {description, order})
      -[:IMPOSED]->                (Sanction {type, months, target_role})
      -[:CITES]->                  (Law {name})
  (Cause)-[:OF_TYPE]->             (CauseCategory {name})
  (Cause)-[:LEADS_TO]->            (Cause)      # causal chain within an accident

The category layer is what makes cross-document aggregation possible:
every accident's free-text causes are pinned to a shared taxonomy.

Also chunks each verdict's text into (Accident)-[:HAS_CHUNK]->(AContent
{chunk, embedding}) nodes with an `accident_chunk_index` vector index, so the
application's retrievers run directly on the real corpus.

Usage: python graph/build_accident_graph.py [--no-embed]
"""
import argparse
import json
import os

import neo4j
from dotenv import load_dotenv

load_dotenv()
URI = os.getenv('NEO4J_URI', 'neo4j://127.0.0.1:7687')
AUTH = ('neo4j', os.getenv('NEO4J_PASSWORD', '12345678'))
IN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'kmst', 'accidents_graph.json')
RAW_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'kmst', 'verdicts_raw.jsonl')
INDEX_NAME = 'accident_chunk_index'

CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Accident) REQUIRE a.verdict_no IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:AContent) REQUIRE c.content_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:CauseCategory) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Court) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Law) REQUIRE l.name IS UNIQUE",
]


def load(session, rec):
    a = rec['accident']
    session.run("""
        MERGE (acc:Accident {verdict_no: $vno})
        SET acc.name = $name, acc.type = $type, acc.date = $date,
            acc.night = $night, acc.weather = $weather,
            acc.human_factors = $hf, acc.keywords = $kw
        MERGE (ct:Court {name: $court})
        MERGE (acc)-[:ADJUDICATED_BY]->(ct)
    """, vno=rec['verdict_no'], name=a.get('name'), type=a.get('type'),
         date=a.get('date'), night=a.get('night'), weather=a.get('weather'),
         hf=rec.get('human_factors', []), kw=rec.get('keywords', []),
         court=rec.get('court', '미상'))

    if a.get('location'):
        session.run("""
            MATCH (acc:Accident {verdict_no: $vno})
            MERGE (loc:ALocation {name: $loc})
            MERGE (acc)-[:OCCURRED_IN]->(loc)
        """, vno=rec['verdict_no'], loc=a['location'])

    for v in rec.get('vessels', []):
        if not v.get('name'):
            continue
        session.run("""
            MATCH (acc:Accident {verdict_no: $vno})
            MERGE (ves:AVessel {name: $name})
            SET ves.type = $type, ves.gross_tonnage = $gt
            MERGE (acc)-[r:INVOLVES]->(ves) SET r.role = $role
        """, vno=rec['verdict_no'], name=v['name'], type=v.get('type'),
             gt=v.get('gross_tonnage'), role=v.get('role'))

    prev_id = None
    for c in sorted(rec.get('cause_chain', []), key=lambda x: x.get('order', 0)):
        res = session.run("""
            MATCH (acc:Accident {verdict_no: $vno})
            CREATE (cause:Cause {description: $desc, order: $order})
            MERGE (cat:CauseCategory {name: $cat})
            MERGE (cause)-[:OF_TYPE]->(cat)
            MERGE (acc)-[:HAS_CAUSE]->(cause)
            RETURN elementId(cause) AS id
        """, vno=rec['verdict_no'], desc=c.get('description'),
             order=c.get('order', 0), cat=c.get('category', '기타'))
        cid = res.single()['id']
        if prev_id:
            session.run("""
                MATCH (p:Cause) WHERE elementId(p) = $p
                MATCH (n:Cause) WHERE elementId(n) = $n
                MERGE (p)-[:LEADS_TO]->(n)
            """, p=prev_id, n=cid)
        prev_id = cid

    for s in rec.get('sanctions', []):
        session.run("""
            MATCH (acc:Accident {verdict_no: $vno})
            CREATE (sa:Sanction {type: $type, months: $months, target_role: $role})
            MERGE (acc)-[:IMPOSED]->(sa)
        """, vno=rec['verdict_no'], type=s.get('type'),
             months=s.get('months'), role=s.get('target_role'))

    for law in rec.get('laws', []):
        session.run("""
            MATCH (acc:Accident {verdict_no: $vno})
            MERGE (l:Law {name: $law})
            MERGE (acc)-[:CITES]->(l)
        """, vno=rec['verdict_no'], law=law)


def chunk_text(text, size=600, overlap=60):
    text = ' '.join(str(text).split())
    return [text[i:i + size] for i in range(0, len(text), size - overlap)
            if text[i:i + size].strip()][:6]


def load_chunks(session):
    """Verdict text chunks attached to accidents (application retrieval layer)."""
    if not os.path.exists(RAW_PATH):
        print('verdicts_raw.jsonl not found - skipping text chunks')
        return
    n = 0
    with open(RAW_PATH, encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get('summary_text') or (rec.get('full_text') or '')[:4000]
            if not text:
                continue
            for j, chunk in enumerate(chunk_text(text)):
                session.run("""
                    MATCH (a:Accident {verdict_no: $vno})
                    MERGE (c:AContent {content_id: $cid})
                    SET c.chunk = $chunk
                    MERGE (a)-[:HAS_CHUNK]->(c)
                """, vno=rec['verdict_no'], cid=f"{rec['verdict_no']}#{j}", chunk=chunk)
                n += 1
    print(f'text chunks: {n}')


def embed_chunks(driver):
    from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
    from neo4j_graphrag.indexes import create_vector_index
    embedder = OpenAIEmbeddings(model='text-embedding-3-small')
    with driver.session() as session:
        records = session.run(
            "MATCH (c:AContent) WHERE c.embedding IS NULL "
            "RETURN elementId(c) AS id, c.chunk AS text").data()
        print(f'embedding {len(records)} accident chunks...')
        for rec in records:
            vector = embedder.embed_query(rec['text'])
            if hasattr(vector, 'tolist'):
                vector = vector.tolist()
            session.run("MATCH (c) WHERE elementId(c) = $id SET c.embedding = $v",
                        id=rec['id'], v=vector)
    create_vector_index(driver, INDEX_NAME, label='AContent',
                        embedding_property='embedding',
                        dimensions=1536, similarity_fn='cosine')
    print('accident vector index ensured.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-embed', action='store_true')
    args = ap.parse_args()
    with open(IN_PATH, encoding='utf-8') as f:
        records = json.load(f)

    driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
    with driver.session() as session:
        # wipe only the accident layer
        session.run("MATCH (n) WHERE any(l IN labels(n) WHERE l IN "
                    "['Accident','AVessel','ALocation','Court','Cause',"
                    "'CauseCategory','Sanction','Law','AContent']) DETACH DELETE n")
        for c in CONSTRAINTS:
            session.run(c)
        for rec in records:
            load(session, rec)
        load_chunks(session)

        counts = session.run(
            "MATCH (n) WHERE any(l IN labels(n) WHERE l IN "
            "['Accident','AVessel','ALocation','Court','Cause','CauseCategory','Sanction','Law','AContent']) "
            "RETURN labels(n)[0] AS l, count(*) AS n ORDER BY l").data()
    print('accident layer:', {c['l']: c['n'] for c in counts})
    if not args.no_embed:
        embed_chunks(driver)
    driver.close()
    print('ACCIDENT_GRAPH_OK')


if __name__ == '__main__':
    main()
