"""
Parse manually-downloaded KMST verdict files (data/kmst/manual/) into the
same verdicts_raw.jsonl format the web fetcher produces.

Handles three formats the tribunal publishes:
  .pdf   -> pypdf text extraction
  .hwpx  -> zip container of section XML: strip tags
  .hwp   -> HWP 5.x OLE binary: decompress BodyText/Section* streams and
            decode UTF-16 paragraph text records (falls back to PrvText preview)

The verdict number is recovered from the filename
(e.g. "인천해심 제2025-017호 ..." -> IC2025-017); records are deduped against
whatever the web fetcher already collected.

Usage: python ingest/parse_manual_verdicts.py
"""
import io
import json
import os
import re
import struct
import zipfile
import zlib

import unicodedata

import olefile
from pypdf import PdfReader

MANUAL_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'kmst', 'manual')
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'kmst', 'verdicts_raw.jsonl')

COURT_CODES = {'인천해심': 'IC', '부산해심': 'BS', '목포해심': 'MP',
               '동해해심': 'DH', '중앙해심': 'JA', '제주해심': 'JJ'}


# ---------------- text extraction per format ----------------
def text_from_pdf(path):
    reader = PdfReader(path)
    return '\n'.join(p.extract_text() or '' for p in reader.pages)


def text_from_hwpx(path):
    out = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if re.match(r'Contents/section\d+\.xml', name):
                xml = z.read(name).decode('utf-8', 'ignore')
                # paragraph text lives in <hp:t> elements
                texts = re.findall(r'<hp:t[^>]*>([^<]*)</hp:t>', xml)
                out.append(' '.join(texts))
    return '\n'.join(out)


def text_from_hwp(path):
    """HWP 5.x: decompress BodyText sections, pull HWPTAG_PARA_TEXT records."""
    ole = olefile.OleFileIO(path)
    try:
        # compression flag in FileHeader
        header = ole.openstream('FileHeader').read()
        compressed = bool(header[36] & 1)
        out = []
        for entry in sorted(ole.listdir()):
            if entry[0] != 'BodyText':
                continue
            data = ole.openstream(entry).read()
            if compressed:
                try:
                    data = zlib.decompress(data, -15)
                except zlib.error:
                    continue
            i = 0
            while i + 4 <= len(data):
                (hdr,) = struct.unpack('<I', data[i:i + 4])
                tag, size = hdr & 0x3FF, (hdr >> 20) & 0xFFF
                i += 4
                if size == 0xFFF:  # extended size
                    (size,) = struct.unpack('<I', data[i:i + 4])
                    i += 4
                if tag == 67:  # HWPTAG_PARA_TEXT
                    chunk = data[i:i + size]
                    chars = []
                    j = 0
                    while j + 1 < len(chunk):
                        (code,) = struct.unpack('<H', chunk[j:j + 2])
                        if code >= 32:
                            chars.append(chr(code))
                            j += 2
                        elif code in (10, 13):
                            chars.append('\n')
                            j += 2
                        elif code in (1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23):
                            j += 16  # inline controls carry 7 extra WCHARs
                        else:
                            j += 2
                    out.append(''.join(chars))
                i += size
        text = '\n'.join(out)
        if len(text) > 150:
            return text
        # fallback: preview stream
        if ole.exists('PrvText'):
            return ole.openstream('PrvText').read().decode('utf-16-le', 'ignore')
        return text
    finally:
        ole.close()


# ---------------- filename -> verdict identity ----------------
COURT_IN_TEXT = {'인천지방해양안전심판원': 'IC', '부산지방해양안전심판원': 'BS',
                 '목포지방해양안전심판원': 'MP', '동해지방해양안전심판원': 'DH',
                 '제주지방해양안전심판원': 'JJ', '중앙해양안전심판원': 'JA'}


def identify(fname, text=''):
    fname = unicodedata.normalize('NFC', fname)
    court = next((COURT_CODES[c] for c in COURT_CODES if c in fname), None)
    if not court and text:
        text_nfc = unicodedata.normalize('NFC', text[:3000])
        court = next((v for k, v in COURT_IN_TEXT.items() if k in text_nfc), None)
    m_no = re.search(r'(\d{4})\s*-\s*(\d{1,3})', fname)
    if not m_no:
        return None, None
    year, num = m_no.group(1), int(m_no.group(2))
    code = court or 'XX'
    return f'{code}{year}-{num:03d}', (code, year, num)


def existing_keys():
    keys = set()
    if not os.path.exists(OUT_PATH):
        return keys
    with open(OUT_PATH, encoding='utf-8') as f:
        for line in f:
            vno = json.loads(line)['verdict_no']
            m = re.match(r'([A-Z]{2})(\d{4})-?(\d+)', vno)
            if m:
                keys.add((m.group(1), m.group(2), int(m.group(3))))
    return keys


def main():
    seen = existing_keys()
    print(f'existing corpus keys: {len(seen)}')

    cases = {}
    skipped = []
    for fname in sorted(os.listdir(MANUAL_DIR)):
        if fname.startswith('._'):
            continue
        path = os.path.join(MANUAL_DIR, fname)
        if not os.path.isfile(path):
            continue
        try:
            if fname.lower().endswith('.pdf'):
                text = text_from_pdf(path)
            elif fname.lower().endswith('.hwpx'):
                text = text_from_hwpx(path)
            elif fname.lower().endswith('.hwp'):
                text = text_from_hwp(path)
            else:
                skipped.append(fname)
                continue
        except Exception as e:
            print(f'  parse failed {fname[:50]}: {e}')
            continue
        if not text or len(text) < 200:
            print(f'  too short ({len(text or "")}) {fname[:50]}')
            continue
        text = unicodedata.normalize('NFC', text)
        vno, key = identify(fname, text)
        if not vno:
            skipped.append(fname)
            continue
        court_names = {v: k[:2] for k, v in COURT_CODES.items()}
        c = cases.setdefault(vno, {'verdict_no': vno, 'key': key,
                                   'court': court_names.get(key[0], '미상'),
                                   'files': [], 'summary_text': None, 'full_text': None})
        c['files'].append(fname)
        if '요약' in unicodedata.normalize('NFC', fname):
            c['summary_text'] = text
        else:
            # keep the longest full text if multiple
            if not c['full_text'] or len(text) > len(c['full_text']):
                c['full_text'] = text

    n_new = 0
    with open(OUT_PATH, 'a', encoding='utf-8') as f:
        for vno, c in cases.items():
            if c['key'] in seen:
                continue
            key = c.pop('key')
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
            seen.add(key)
            n_new += 1
    print(f'manual files parsed: {sum(len(c["files"]) for c in cases.values())}, '
          f'cases: {len(cases)}, new appended: {n_new}, unidentified: {len(skipped)}')
    for s in skipped[:5]:
        print('  skipped:', s[:60])


if __name__ == '__main__':
    main()
