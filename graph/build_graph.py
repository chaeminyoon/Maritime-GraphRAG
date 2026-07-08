"""
Build the maritime knowledge graph in Neo4j.

Two layers:
  Document layer   (Article)-[:HAS_CHUNK]->(Content{embedding})
                   (Article)-[:PUBLISHED_BY]->(Media), (Article)-[:BELONGS_TO]->(Category)
  Knowledge layer  (Company)-[:OPERATES]->(Vessel{type})
                   (Vessel)-[:CALLS_AT]->(Port)
                   (Regulation)-[:APPLIES_TO]->(Vessel)   # materialized per vessel of the type
                   (Vessel)-[:INVOLVED_IN]->(Incident)-[:OCCURRED_AT]->(Port)
                   (Article)-[:MENTIONS]->(Vessel|Company|Port|Regulation|Incident)

For the bundled synthetic corpus the knowledge layer is loaded from
data/corpus/entities.json (relations known by construction — that is what makes
the retrieval benchmark scoreable). For real documents, use
ingest/extract_entities.py to produce the same structure with an LLM.

Embeddings (OpenAI text-embedding-3-small) are written to Content nodes at build
time; a vector index is created over them.

Usage:  python graph/build_graph.py [--no-embed]
"""
import argparse
import json
import os
import sys

import neo4j
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "12345678"))
INDEX_NAME = "content_vector_index"
DIMENSION = 1536
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'corpus')

CHUNK_SIZE, OVERLAP = 500, 50


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    text = str(text)
    return [text[i:i + chunk_size].strip()
            for i in range(0, len(text), chunk_size - overlap)
            if text[i:i + chunk_size].strip()]


CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.article_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Content) REQUIRE c.content_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Media) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (cat:Category) REQUIRE cat.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Vessel) REQUIRE v.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (co:Company) REQUIRE co.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Port) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Regulation) REQUIRE r.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Incident) REQUIRE i.incident_id IS UNIQUE",
]


def build_knowledge_layer(session, entities):
    for name in entities['companies']:
        session.run("MERGE (:Company {name: $n})", n=name)
    for p in entities['ports']:
        session.run("MERGE (:Port {name: $n})", n=p)
    for v in entities['vessels']:
        session.run("""
            MERGE (ves:Vessel {name: $name}) SET ves.type = $type
            MERGE (co:Company {name: $op})
            MERGE (co)-[:OPERATES]->(ves)
        """, name=v['name'], type=v['type'], op=v['operator'])
    for vessel, ports in entities['calls_at'].items():
        for p in ports:
            session.run("""
                MATCH (v:Vessel {name: $v}), (p:Port {name: $p})
                MERGE (v)-[:CALLS_AT]->(p)
            """, v=vessel, p=p)
    for r in entities['regulations']:
        session.run("""
            MERGE (reg:Regulation {name: $n})
            SET reg.applies_to_type = $t, reg.description = $d
        """, n=r['name'], t=r['applies_to'], d=r['description'])
        session.run("""
            MATCH (reg:Regulation {name: $n}), (v:Vessel {type: $t})
            MERGE (reg)-[:APPLIES_TO]->(v)
        """, n=r['name'], t=r['applies_to'])
    for i in entities['incidents']:
        session.run("""
            MERGE (inc:Incident {incident_id: $id}) SET inc.description = $d
            WITH inc
            MATCH (v:Vessel {name: $v}), (p:Port {name: $p})
            MERGE (v)-[:INVOLVED_IN]->(inc)
            MERGE (inc)-[:OCCURRED_AT]->(p)
        """, id=i['id'], d=i['description'], v=i['vessel'], p=i['port'])


def entity_names(entities):
    """(name, label, match_key) for MENTIONS scanning."""
    out = [(c, 'Company', 'name') for c in entities['companies']]
    out += [(p, 'Port', 'name') for p in entities['ports']]
    out += [(v['name'], 'Vessel', 'name') for v in entities['vessels']]
    out += [(r['name'], 'Regulation', 'name') for r in entities['regulations']]
    return out


def build_document_layer(session, df, entities):
    names = entity_names(entities)
    for _, row in df.iterrows():
        session.run("""
            MERGE (a:Article {article_id: $id})
            SET a.title = $title, a.url = $url, a.published_date = $date
            MERGE (m:Media {name: $source})
            MERGE (a)-[:PUBLISHED_BY]->(m)
            MERGE (c:Category {name: $category})
            MERGE (a)-[:BELONGS_TO]->(c)
        """, id=row['article_id'], title=row['title'], url=row['url'],
             date=row['published_date'], source=row['source'], category=row['category'])

        for j, chunk in enumerate(chunk_text(row['content'])):
            session.run("""
                MATCH (a:Article {article_id: $id})
                MERGE (c:Content {content_id: $cid})
                SET c.chunk = $chunk
                MERGE (a)-[:HAS_CHUNK]->(c)
            """, id=row['article_id'], cid=f"{row['article_id']}-{j}", chunk=chunk)

        # MENTIONS: scan title+content for known entity names
        text = f"{row['title']} {row['content']}"
        for name, label, key in names:
            if name in text:
                session.run(f"""
                    MATCH (a:Article {{article_id: $id}}), (e:{label} {{{key}: $name}})
                    MERGE (a)-[:MENTIONS]->(e)
                """, id=row['article_id'], name=name)
        # incident mentions via relations ground truth
        for rel in json.loads(row['relations']):
            if rel[0] == 'INVOLVED_IN':
                session.run("""
                    MATCH (a:Article {article_id: $id}), (i:Incident {incident_id: $iid})
                    MERGE (a)-[:MENTIONS]->(i)
                """, id=row['article_id'], iid=rel[2])


def embed_contents(driver):
    from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
    from neo4j_graphrag.indexes import create_vector_index

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    with driver.session() as session:
        records = session.run(
            "MATCH (c:Content) WHERE c.embedding IS NULL "
            "RETURN elementId(c) AS id, c.chunk AS text").data()
        print(f"embedding {len(records)} chunks...")
        for rec in records:
            vector = embedder.embed_query(rec['text'])
            if hasattr(vector, 'tolist'):
                vector = vector.tolist()
            session.run("MATCH (c) WHERE elementId(c) = $id SET c.embedding = $v",
                        id=rec['id'], v=vector)
    create_vector_index(driver, INDEX_NAME, label="Content",
                        embedding_property="embedding",
                        dimensions=DIMENSION, similarity_fn="cosine")
    print("vector index ensured.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-embed', action='store_true',
                    help='skip OpenAI embedding generation')
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(DATA_DIR, 'maritime_corpus.csv'))
    with open(os.path.join(DATA_DIR, 'entities.json'), encoding='utf-8') as f:
        entities = json.load(f)

    driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"Neo4j connection failed: {e}")
        sys.exit(1)

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        for c in CONSTRAINTS:
            session.run(c)
        build_knowledge_layer(session, entities)
        build_document_layer(session, df, entities)

        counts = session.run("""
            MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label
        """).data()
        rels = session.run("""
            MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY type
        """).data()
    print("nodes:", {c['label']: c['n'] for c in counts})
    print("rels :", {r['type']: r['n'] for r in rels})

    if not args.no_embed:
        embed_contents(driver)
    driver.close()
    print("GRAPH_BUILD_OK")


if __name__ == '__main__':
    main()
