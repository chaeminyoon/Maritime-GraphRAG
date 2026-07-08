"""
LLM entity/relation extractor for REAL maritime documents.

The bundled synthetic corpus ships with ground-truth relations, so
graph/build_graph.py loads them directly. For real articles (press releases,
industry news) this module produces the same structure with gpt-4o:

    {"vessels": [{"name", "type", "operator"}],
     "companies": [...], "ports": [...],
     "calls_at": {vessel: [ports]},
     "regulations": [{"name", "applies_to", "description"}],
     "incidents": [{"id", "vessel", "port", "description"}]}

Usage:
    python ingest/extract_entities.py --csv my_articles.csv --out data/corpus/entities.json
    (csv needs 'article_id' and 'content' columns; merge/dedupe is naive by name)

Extraction quality is NOT benchmarked here — the retrieval benchmark uses the
synthetic corpus precisely to isolate retrieval quality from extraction noise.
"""
import argparse
import json

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

PROMPT = """다음 해양 산업 기사에서 엔티티와 관계를 추출해 JSON으로 반환하라.
스키마:
{
  "vessels": [{"name": str, "type": "컨테이너선|유조선|LNG운반선|벌크선|기타", "operator": str|null}],
  "companies": [str], "ports": [str],
  "calls_at": {"<선박명>": ["<항만>"]},
  "regulations": [{"name": str, "applies_to": str, "description": str}],
  "incidents": [{"vessel": str, "port": str, "description": str}]
}
기사에 명시된 것만 추출하고, 추측하지 마라. 해당 없으면 빈 배열/객체.

[기사]
{text}
"""


def extract(text):
    resp = client.chat.completions.create(
        model="gpt-4o", temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": PROMPT.replace('{text}', text[:6000])}])
    return json.loads(resp.choices[0].message.content)


def merge(agg, ext, article_id):
    for c in ext.get('companies', []):
        if c not in agg['companies']:
            agg['companies'].append(c)
    for p in ext.get('ports', []):
        if p not in agg['ports']:
            agg['ports'].append(p)
    known = {v['name'] for v in agg['vessels']}
    for v in ext.get('vessels', []):
        if v.get('name') and v['name'] not in known:
            agg['vessels'].append({'name': v['name'],
                                   'type': v.get('type', '기타'),
                                   'operator': v.get('operator') or '미상'})
    for vessel, ports in ext.get('calls_at', {}).items():
        agg['calls_at'].setdefault(vessel, [])
        for p in ports:
            if p not in agg['calls_at'][vessel]:
                agg['calls_at'][vessel].append(p)
    known_r = {r['name'] for r in agg['regulations']}
    for r in ext.get('regulations', []):
        if r.get('name') and r['name'] not in known_r:
            agg['regulations'].append(r)
    for i, inc in enumerate(ext.get('incidents', [])):
        inc['id'] = f"{article_id}-inc{i}"
        agg['incidents'].append(inc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help="articles csv (article_id, content)")
    ap.add_argument('--out', default='data/corpus/entities.json')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    agg = {'companies': [], 'vessels': [], 'ports': [], 'calls_at': {},
           'regulations': [], 'incidents': []}
    for _, row in df.iterrows():
        try:
            ext = extract(str(row['content']))
            merge(agg, ext, row['article_id'])
            print(f"{row['article_id']}: ok")
        except Exception as e:
            print(f"{row['article_id']}: extraction failed ({e})")

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(f"saved {args.out}: {len(agg['vessels'])} vessels, "
          f"{len(agg['companies'])} companies, {len(agg['incidents'])} incidents")


if __name__ == '__main__':
    main()
