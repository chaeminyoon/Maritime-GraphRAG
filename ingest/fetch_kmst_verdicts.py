"""
Fetch Korean Maritime Safety Tribunal (KMST) verdict documents.

The KMST publishes adjudication reports (재결서) for every investigated marine
casualty — public-sector works under the KOGL (공공누리) license. Each listing
row links two PDFs: the full verdict (공개용) and a structured summary
(재결요약서: 판시요지, causes, laws, keywords, vessel table).

This fetcher pulls the latest N rows, downloads both PDFs per case, extracts
text (pypdf), and writes data/kmst/verdicts_raw.jsonl with one record per case:
    {verdict_no, court, files: [names], summary_text, full_text}

Politeness: sequential, 0.4 s delay, identified User-Agent.
Raw PDFs/texts are NOT committed to git (see .gitignore) — rerun this script
to reproduce; the committed artifact is the LLM-extracted structured graph
(data/kmst/accidents_graph.json).

Usage: python ingest/fetch_kmst_verdicts.py --n 100
"""
import argparse
import io
import json
import os
import re
import time
import urllib.parse

import requests
from pypdf import PdfReader

BASE = 'https://www.kmst.go.kr'
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'kmst')

COURTS = {'BS': '부산', 'IC': '인천', 'MP': '목포', 'DH': '동해', 'JA': '중앙', 'JJ': '제주'}


def fetch_list(session, n):
    cases = {}
    for page in range(1, 20):
        r = session.post(f'{BASE}/web/verdictList.do',
                         data={'menuIdx': 121, 'pageindex': page,
                               'recordCountPerPage': 25, 'orgn_cd': '',
                               'st': '', 'searchWord': ''},
                         timeout=60)
        r.raise_for_status()
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', r.text, re.S)
        found = 0
        for row in rows:
            pairs = re.findall(r"downloadVerdict\('(\d+)'\s*,\s*'([A-Z]{2}\d+)'\)", row)
            if not pairs:
                continue
            verdict_no = pairs[0][1]
            if verdict_no not in cases:
                cases[verdict_no] = [p[0] for p in pairs]
                found += 1
            if len(cases) >= n:
                return cases
        print(f'  page {page}: +{found} new (total {len(cases)})')
        if found == 0:   # no new rows -> end of list
            return cases
        time.sleep(0.4)
    return cases


def download_pdf_text(session, atch_id, retries=2):
    for attempt in range(retries + 1):
        r = session.get(f'{BASE}/web/atch/atchFileDownload.do',
                        params={'atchId': atch_id, 'fileSn': 1}, timeout=90)
        # older files are served as application/octet-stream -> sniff magic bytes
        if r.status_code == 200 and r.content.startswith(b'%PDF'):
            break
        print(f'    atch {atch_id}: blocked (status {r.status_code}, '
              f'{r.headers.get("Content-Type", "?")[:40]}), '
              f'retry {attempt + 1}/{retries} after pause')
        time.sleep(15)
        session.cookies.clear()
    else:
        return None, None
    header = r.headers.get('Content-Disposition', '') + r.headers.get('Content-Type', '')
    m = re.search(r'(?:filename|name)="?([^";]+)', header)
    fname = urllib.parse.unquote(m.group(1).replace('+', ' ')) if m else f'{atch_id}.pdf'
    try:
        reader = PdfReader(io.BytesIO(r.content))
        text = '\n'.join(p.extract_text() or '' for p in reader.pages)
    except Exception:
        return fname, None
    return fname, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100, help='number of cases')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0'

    cases = fetch_list(session, args.n)
    print(f'listed {len(cases)} cases')

    out_path = os.path.join(OUT_DIR, 'verdicts_raw.jsonl')
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding='utf-8') as f:
            done = {json.loads(l)['verdict_no'] for l in f if l.strip()}
        print(f'resuming: {len(done)} already fetched')

    n_ok = len(done)
    n_new = 0
    with open(out_path, 'a', encoding='utf-8') as f:
        for verdict_no, atch_ids in cases.items():
            if verdict_no in done:
                continue
            rec = {'verdict_no': verdict_no,
                   'court': COURTS.get(verdict_no[:2], verdict_no[:2]),
                   'files': [], 'summary_text': None, 'full_text': None}
            n_new += 1
            if n_new % 15 == 0:   # rotate session to dodge rate limiting
                session = requests.Session()
                session.headers['User-Agent'] = 'Mozilla/5.0'
                time.sleep(5)
            for atch_id in atch_ids[:2]:
                fname, text = download_pdf_text(session, atch_id)
                time.sleep(1.2)
                if not text:
                    continue
                rec['files'].append(fname)
                if '재결요약서' in (fname or '') or '요약' in (fname or ''):
                    rec['summary_text'] = text
                else:
                    rec['full_text'] = text
            if rec['summary_text'] or rec['full_text']:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                n_ok += 1
                print(f'  [{n_ok}] {verdict_no} ({rec["court"]}) files={len(rec["files"])}')
    print(f'saved {n_ok} cases -> {out_path}')


if __name__ == '__main__':
    main()
