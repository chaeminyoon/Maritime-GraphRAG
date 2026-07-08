"""
LLM extraction: KMST verdict text -> structured accident graph records.

Input : data/kmst/verdicts_raw.jsonl   (from ingest/fetch_kmst_verdicts.py)
Output: data/kmst/accidents_graph.json (committed — the reproducible artifact)

Each verdict summary (재결요약서) states the causal findings for ONE accident.
The extraction normalizes free-text causes onto a fixed category taxonomy
(modeled on the KMST cause-code hierarchy) so that cause chains become
aggregatable across hundreds of documents — the whole point of the graph.

Resumable: already-extracted verdict_no are skipped on rerun.
"""
import argparse
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

IN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'kmst', 'verdicts_raw.jsonl')
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'kmst', 'accidents_graph.json')

CAUSE_CATEGORIES = [
    '경계 소홀', '항행법규 위반', '조선 부적절', '위치확인 소홀',
    '정비·점검 소홀', '기기취급 불량', '작업안전수칙 미준수', '화기취급 부주의',
    '적재·고박 불량', '기상·해상 불량', '선체·설비 결함', '음주·복무기강 위반',
    '안전관리체제 미흡', '무리한 출항·운항', '기타',
]
ACCIDENT_TYPES = ['충돌', '접촉', '좌초', '전복', '침몰', '화재·폭발', '기관손상',
                  '해양오염', '인명사상', '침수', '운항저해', '기타']
VESSEL_TYPES = ['어선', '화물선', '유조선', '여객선', '예인선', '부선', '수상레저기구', '기타']

PROMPT = """다음은 해양안전심판원 재결(요약)서 텍스트다. 사고 정보를 추출해 JSON으로 반환하라.

스키마 (반드시 이 키들만):
{
  "accident": {
    "name": "사건명",
    "type": "%(types)s 중 하나",
    "date": "YYYY-MM-DD 또는 null",
    "night": "야간(일몰~일출) 발생이면 true, 주간이면 false, 판단 불가면 null",
    "location": "사고 장소 (해역/항만명, 간결히)",
    "weather": "기상·시계 요약 (예: '안개, 시계 0.5마일') 또는 null"
  },
  "vessels": [{"name": "선명", "type": "%(vtypes)s 중 하나", "gross_tonnage": 숫자 또는 null, "role": "가해/피해/단독 등"}],
  "cause_chain": [
    {"order": 1, "description": "원인 서술 (간결히)", "category": "%(cats)s 중 하나"}
  ],
  "human_factors": ["1인 당직", "피로", "음주" 등 인적 요인, 없으면 []],
  "sanctions": [{"target_role": "선장/기관장/사업자 등", "type": "업무정지/견책/시정권고/개선권고 등", "months": 숫자 또는 null}],
  "laws": ["관련 법규"],
  "keywords": ["주제어"]
}

규칙:
- cause_chain은 재결이 판시한 원인을 인과 순서대로(선행 요인 -> 직접 원인) 나열하라.
- category는 반드시 주어진 목록에서 골라라. 텍스트에 명시된 것만 추출하고 추측하지 마라.

[재결서 텍스트]
%(text)s
"""


def extract(text):
    prompt = PROMPT % dict(types='|'.join(ACCIDENT_TYPES),
                           vtypes='|'.join(VESSEL_TYPES),
                           cats='|'.join(CAUSE_CATEGORIES),
                           text=text[:7000])
    resp = client.chat.completions.create(
        model='gpt-4o', temperature=0,
        response_format={'type': 'json_object'},
        messages=[{'role': 'user', 'content': prompt}])
    return json.loads(resp.choices[0].message.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='max cases (0 = all)')
    args = ap.parse_args()

    done = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding='utf-8') as f:
            done = {r['verdict_no']: r for r in json.load(f)}
        print(f'resuming: {len(done)} already extracted')

    records = list(done.values())
    n = 0
    with open(IN_PATH, encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            vno = rec['verdict_no']
            if vno in done:
                continue
            text = rec.get('summary_text') or rec.get('full_text')
            if not text or len(text) < 200:
                continue
            try:
                ext = extract(text)
            except Exception as e:
                print(f'  {vno}: FAILED ({e})')
                continue
            ext['verdict_no'] = vno
            ext['court'] = rec['court']
            records.append(ext)
            n += 1
            cats = [c.get('category') for c in ext.get('cause_chain', [])]
            print(f'  [{len(records)}] {vno} {ext["accident"].get("type")}: {cats}')
            if args.limit and n >= args.limit:
                break
            # checkpoint every 10
            if n % 10 == 0:
                with open(OUT_PATH, 'w', encoding='utf-8') as g:
                    json.dump(records, g, ensure_ascii=False, indent=1)

    with open(OUT_PATH, 'w', encoding='utf-8') as g:
        json.dump(records, g, ensure_ascii=False, indent=1)
    print(f'saved {len(records)} accidents -> {OUT_PATH}')


if __name__ == '__main__':
    main()
