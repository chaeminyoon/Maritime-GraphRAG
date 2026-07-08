"""
Synthetic maritime news corpus + ground-truth QA benchmark generator.

Real maritime news cannot ship with the repo (copyright), and — more importantly —
real news gives no ground truth to score retrieval against. This generator creates
a fictional-but-realistic Korean maritime shipping world where every entity
relation is KNOWN BY CONSTRUCTION:

    Company --OPERATES-->   Vessel (typed: 컨테이너선/유조선/LNG운반선/벌크선)
    Vessel  --CALLS_AT-->   Port
    Regulation --APPLIES_TO--> vessel type
    Vessel  --INVOLVED_IN--> Incident --OCCURRED_AT--> Port

~46 news-style articles express these relations in prose (plus noise articles
with no relations), and a QA benchmark is derived from the relation tables:
single-hop questions answerable from one article, multi-hop questions whose
answer exists in NO single article — the case where graph-aware retrieval
must prove itself against pure vector search.

All entities are fictional; ports and regulation names follow real-world
conventions (부산항, IMO CII 등) for realism. Deterministic (seeded).

Outputs:
    data/corpus/maritime_corpus.csv      one row per article
    data/corpus/entities.json            entity + relation tables (graph ground truth)
    data/benchmark/qa_benchmark.json     QA pairs with gold answer entities
"""
import json
import os
import random
import pandas as pd

SEED = 7
OUT_CORPUS = os.path.join(os.path.dirname(__file__), '..', 'data', 'corpus')
OUT_BENCH = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmark')

# ---------------- fictional world ----------------
COMPANIES = ['한서해운', '대양상선', '청림로지스틱스', '파도글로벌해운', '남명탱커', '동주컨테이너라인']

# (vessel, type, operator)
VESSELS = [
    ('한서파이오니어', '컨테이너선', '한서해운'),
    ('한서오디세이', '컨테이너선', '한서해운'),
    ('한서글로리', '벌크선', '한서해운'),
    ('대양스피릿', 'LNG운반선', '대양상선'),
    ('대양호라이즌', '벌크선', '대양상선'),
    ('대양챔피언', '컨테이너선', '대양상선'),
    ('청림익스프레스', '컨테이너선', '청림로지스틱스'),
    ('청림웨이브', '컨테이너선', '청림로지스틱스'),
    ('파도스타', '벌크선', '파도글로벌해운'),
    ('파도블루', '컨테이너선', '파도글로벌해운'),
    ('남명선라이즈', '유조선', '남명탱커'),
    ('남명퍼시픽', '유조선', '남명탱커'),
    ('동주하모니', '컨테이너선', '동주컨테이너라인'),
    ('동주비전', '컨테이너선', '동주컨테이너라인'),
]

PORTS = ['부산항', '인천항', '광양항', '울산항', '싱가포르항', '로테르담항']

# vessel -> ports it calls at (deterministic routes)
CALLS_AT = {
    '한서파이오니어': ['부산항', '싱가포르항', '로테르담항'],
    '한서오디세이': ['부산항', '인천항'],
    '한서글로리': ['광양항', '싱가포르항'],
    '대양스피릿': ['울산항', '싱가포르항'],
    '대양호라이즌': ['광양항', '로테르담항'],
    '대양챔피언': ['부산항', '로테르담항'],
    '청림익스프레스': ['인천항', '싱가포르항'],
    '청림웨이브': ['부산항', '광양항'],
    '파도스타': ['울산항', '광양항'],
    '파도블루': ['인천항', '로테르담항'],
    '남명선라이즈': ['울산항'],
    '남명퍼시픽': ['울산항', '싱가포르항'],
    '동주하모니': ['부산항', '싱가포르항'],
    '동주비전': ['인천항', '부산항'],
}

# (regulation, applies to vessel type, one-line description)
REGULATIONS = [
    ('IMO 탄소집약도지표(CII) 규제', '컨테이너선', '운항 탄소배출 등급을 매년 평가해 D등급 이하 선박에 개선계획을 요구'),
    ('황산화물(SOx) 배출 규제', '유조선', '연료유 황 함유량을 0.5% 이하로 제한'),
    ('선박평형수관리협약(BWM)', '벌크선', '평형수 처리설비 설치를 의무화'),
    ('LNG 벙커링 안전기준', 'LNG운반선', '항만 내 LNG 연료 공급 작업의 안전 절차를 규정'),
]

# (incident_id, vessel, port, description)
INCIDENTS = [
    ('사고-01', '남명선라이즈', '울산항', '기관 고장으로 예인'),
    ('사고-02', '한서파이오니어', '부산항', '접안 중 안벽 접촉'),
    ('사고-03', '파도스타', '광양항', '하역 중 화물창 침수'),
    ('사고-04', '대양스피릿', '싱가포르항', '벙커링 중 경미한 가스 누출 경보'),
    ('사고-05', '동주하모니', '싱가포르항', '컨테이너 고박 불량으로 2개 유실'),
    ('사고-06', '청림익스프레스', '인천항', '입항 중 어선과 근접 조우로 긴급 변침'),
]

SOURCES = ['해양데일리', '마리타임뉴스', '항만경제신문', '쉬핑저널']
DATES = [f'2026-{m:02d}-{d:02d}' for m in range(1, 7) for d in (5, 14, 23)]


def _writer(rng):
    counter = {'i': 0}

    def add(articles, title, body, category, relations):
        counter['i'] += 1
        articles.append({
            'article_id': f'MN{counter["i"]:03d}',
            'title': title,
            'content': body,
            'source': rng.choice(SOURCES),
            'category': category,
            'published_date': rng.choice(DATES),
            'url': f'https://example.com/maritime/{counter["i"]:03d}',
            'relations': relations,   # ground truth expressed in this article
        })
    return add


def build_articles(rng):
    articles = []
    add = _writer(rng)

    # 1) route articles: one per vessel (OPERATES + CALLS_AT)
    for vessel, vtype, company in VESSELS:
        ports = CALLS_AT[vessel]
        route = ' → '.join(ports)
        body = (
            f"{company}이 운영하는 {vtype} {vessel}호가 {route} 노선에 투입된다. "
            f"{company} 관계자는 \"{ports[0]}을 중심으로 한 기항 일정을 통해 정시성을 확보하겠다\"고 밝혔다. "
            f"{vessel}호는 이번 개편으로 {', '.join(ports)}에 정기 기항하게 된다. "
            f"업계에서는 {vtype} 시장의 수급 상황을 고려한 노선 조정으로 평가하고 있다."
        )
        add(articles, f"{company}, {vessel}호 {ports[0]} 중심 노선 개편",
            body, '항로/노선',
            [('OPERATES', company, vessel)] + [('CALLS_AT', vessel, p) for p in ports])

    # 2) regulation articles (APPLIES_TO) - two per regulation
    for reg, vtype, desc in REGULATIONS:
        affected = [v for v, t, c in VESSELS if t == vtype]
        body = (
            f"{reg}가 올해부터 단계적으로 강화된다. 이 규제는 {desc}하는 내용으로, {vtype}에 적용된다. "
            f"국내 선사들이 운영 중인 {vtype} 선대 전반의 대응 투자가 불가피할 전망이다. "
            f"해운업계는 규제 대응 비용을 운임에 반영하는 방안을 검토 중이다."
        )
        add(articles, f"{reg} 강화… {vtype} 선사 대응 비상",
            body, '규제/환경', [('APPLIES_TO', reg, vtype)])

        picks = affected[:2]
        companies = sorted({c for v, t, c in VESSELS if v in picks})
        body2 = (
            f"{reg} 시행을 앞두고 {' 와 '.join(companies)} 등이 보유 {vtype} 개조 계획을 발표했다. "
            f"대상 선박에는 {', '.join(p + '호' for p in picks)} 등이 포함된다. "
            f"이 규제는 {desc}하며, 미이행 선박은 운항 제한을 받을 수 있다."
        )
        add(articles, f"{vtype} 업계, {reg} 대응 투자 착수",
            body2, '규제/환경', [('APPLIES_TO', reg, vtype)])

    # 3) incident articles (INVOLVED_IN + OCCURRED_AT + OPERATES)
    for iid, vessel, port, desc in INCIDENTS:
        vtype, company = next((t, c) for v, t, c in VESSELS if v == vessel)
        body = (
            f"{port}에서 {company} 소속 {vtype} {vessel}호가 {desc}되는 사고가 발생했다. "
            f"항만 당국은 인명 피해는 없다고 밝혔으며, 정확한 원인을 조사 중이다. "
            f"{company}는 \"안전 점검을 강화하고 재발 방지 대책을 마련하겠다\"고 밝혔다."
        )
        add(articles, f"{port}서 {vessel}호 사고… {desc}",
            body, '사고/안전',
            [('INVOLVED_IN', vessel, iid), ('OCCURRED_AT', iid, port),
             ('OPERATES', company, vessel)])

    # 4) company performance articles (OPERATES 재확인 + noise)
    for company in COMPANIES:
        fleet = [v for v, t, c in VESSELS if c == company]
        flagship = fleet[0]
        growth = rng.choice(['12%', '8%', '15%', '5%'])
        body = (
            f"{company}가 지난 분기 매출이 전년 동기 대비 {growth} 증가했다고 공시했다. "
            f"{company}는 주력선 {flagship}호를 포함해 총 {len(fleet)}척을 운영하고 있다. "
            f"회사 측은 운임 회복과 선대 가동률 개선을 실적 개선의 배경으로 꼽았다."
        )
        add(articles, f"{company}, 분기 실적 {growth} 성장",
            body, '기업/실적',
            [('OPERATES', company, flagship)])

    # 5) port/general noise articles (no relations)
    noise = [
        ("글로벌 컨테이너 운임지수 3주 연속 상승", "글로벌 컨테이너 운임지수가 3주 연속 상승세를 이어갔다. 성수기 물동량 증가와 선복 공급 조절이 배경으로 분석된다. 전문가들은 하반기까지 강세가 이어질 수 있다고 전망했다.", '시황'),
        ("부산항 신항 자동화 터미널 2단계 착공", "부산항 신항에서 완전 자동화 터미널 2단계 공사가 시작됐다. 완공 시 하역 생산성이 크게 개선될 것으로 기대된다. 항만공사는 2028년 개장을 목표로 하고 있다.", '항만/인프라'),
        ("광양항, 배후단지 물류기업 유치 설명회", "광양항 배후단지 활성화를 위한 물류기업 대상 투자 설명회가 열렸다. 세제 혜택과 임대료 감면 방안이 소개됐다.", '항만/인프라'),
        ("선원 인력난 심화… 해기사 양성 확대 논의", "해운업계의 선원 인력난이 심화되면서 해기사 양성 확대 방안이 논의되고 있다. 업계는 처우 개선과 승선 근무 환경 개선이 병행돼야 한다고 지적한다.", '인력/노동'),
        ("친환경 선박 연료 시장, 메탄올 주목", "차세대 선박 연료로 메탄올이 주목받고 있다. 기존 연료 대비 배출 저감 효과가 크고 취급이 상대적으로 용이하다는 평가다.", '기술/친환경'),
        ("인천항 크루즈 터미널 이용객 회복세", "인천항 크루즈 터미널의 이용객이 회복세를 보이고 있다. 항만공사는 국제 크루즈 노선 유치를 확대할 계획이다.", '항만/인프라'),
        ("해상보험료 상승세… 사고 증가 영향", "최근 해상 사고 증가로 선박 보험료가 상승세를 보이고 있다. 보험업계는 안전 관리 우수 선사에 대한 요율 차등화를 검토 중이다.", '시황'),
        ("울산항, 액체화물 처리량 역대 최대", "울산항의 연간 액체화물 처리량이 역대 최대치를 기록했다. 항만공사는 저장 인프라 증설을 추진하기로 했다.", '항만/인프라'),
    ]
    for title, body, cat in noise:
        add(articles, title, body, cat, [])

    return articles


def build_benchmark():
    """QA pairs derived from the relation tables (not from article text)."""
    qa = []
    vinfo = {v: (t, c) for v, t, c in VESSELS}

    def add(q, gold, hops, note=''):
        qa.append({'question': q, 'gold_entities': sorted(gold), 'hops': hops, 'note': note})

    # ---- single-hop (answer stated in one article) ----
    add('한서파이오니어호를 운영하는 선사는 어디인가?', ['한서해운'], 1)
    add('대양스피릿호는 어떤 종류의 선박인가?', ['LNG운반선'], 1)
    add('남명선라이즈호의 사고는 어느 항만에서 발생했나?', ['울산항'], 1)
    add('선박평형수관리협약(BWM)은 어떤 선종에 적용되나?', ['벌크선'], 1)
    add('청림익스프레스호가 기항하는 항만은 어디인가?', ['인천항', '싱가포르항'], 1)
    add('부산항에서 접안 중 사고를 낸 선박은?', ['한서파이오니어'], 1)
    add('황산화물(SOx) 배출 규제의 적용 대상 선종은?', ['유조선'], 1)
    add('동주컨테이너라인이 운영하는 선박을 모두 나열하라.', ['동주하모니', '동주비전'], 1)

    # ---- multi-hop (no single article contains the answer) ----
    def vessels_by_type(t):
        return [v for v, (vt, c) in vinfo.items() if vt == t]

    def calls(port):
        return [v for v, ps in CALLS_AT.items() if port in ps]

    # 2-hop: regulation -> type -> vessels calling at a port
    cii_vessels = [v for v in vessels_by_type('컨테이너선') if '로테르담항' in CALLS_AT[v]]
    add('IMO 탄소집약도지표(CII) 규제의 영향을 받는 선박 중 로테르담항에 기항하는 선박은?',
        cii_vessels, 2, 'APPLIES_TO -> vessel type -> CALLS_AT')

    # 2-hop: port -> vessels -> operators (container only)
    busan_ops = sorted({vinfo[v][1] for v in calls('부산항') if vinfo[v][0] == '컨테이너선'})
    add('부산항에 컨테이너선을 기항시키는 선사를 모두 나열하라.',
        busan_ops, 2, 'CALLS_AT -> type filter -> OPERATES')

    # 2-hop: incidents -> vessels -> operators
    incident_ops = sorted({vinfo[v][1] for _, v, _, _ in INCIDENTS})
    add('사고 이력이 있는 선박을 운영하는 선사를 모두 나열하라.',
        incident_ops, 2, 'INVOLVED_IN -> OPERATES')

    # 3-hop: port incident -> vessel -> type -> regulation
    ulsan_vessels = {v for _, v, p, _ in INCIDENTS if p == '울산항'}
    ulsan_regs = sorted({r for r, t, _ in REGULATIONS
                         if t in {vinfo[v][0] for v in ulsan_vessels}})
    add('울산항에서 사고를 낸 선박에 적용되는 환경 규제는 무엇인가?',
        ulsan_regs, 3, 'OCCURRED_AT -> vessel -> type -> APPLIES_TO')

    # 2-hop: company -> vessels -> ports (fleet coverage)
    hanseo_ports = sorted({p for v, (t, c) in vinfo.items() if c == '한서해운'
                           for p in CALLS_AT[v]})
    add('한서해운 선대가 기항하는 항만을 모두 나열하라.',
        hanseo_ports, 2, 'OPERATES -> CALLS_AT')

    # 2-hop: singapore incidents -> operators
    sg_ops = sorted({vinfo[v][1] for _, v, p, _ in INCIDENTS if p == '싱가포르항'})
    add('싱가포르항에서 사고가 발생한 선박들의 운영 선사는?',
        sg_ops, 2, 'OCCURRED_AT -> vessel -> OPERATES')

    # 2-hop: LNG regulation -> vessels -> operator
    lng_ops = sorted({vinfo[v][1] for v in vessels_by_type('LNG운반선')})
    add('LNG 벙커링 안전기준의 적용을 받는 선박을 운영하는 선사는?',
        lng_ops, 2, 'APPLIES_TO -> type -> OPERATES')

    # 2-hop: two regulations on same company's fleet
    pado_types = {vinfo[v][0] for v, (t, c) in vinfo.items() if c == '파도글로벌해운'}
    pado_regs = sorted({r for r, t, _ in REGULATIONS if t in pado_types})
    add('파도글로벌해운 선대에 적용되는 환경 규제를 모두 나열하라.',
        pado_regs, 2, 'OPERATES -> type -> APPLIES_TO')

    # ---- aggregation ----
    add('대양상선이 운영하는 선박은 총 몇 척인가?',
        [str(len([v for v, (t, c) in vinfo.items() if c == '대양상선']))], 1, 'count')
    add('울산항에서 발생한 사고는 총 몇 건인가?',
        [str(len([1 for _, _, p, _ in INCIDENTS if p == '울산항']))], 1, 'count')
    add('컨테이너선은 전체 선대에서 몇 척인가?',
        [str(len(vessels_by_type('컨테이너선')))], 1, 'count')
    add('가장 많은 선박이 기항하는 항만은 어디인가?',
        [max(PORTS, key=lambda p: len(calls(p)))], 2, 'argmax over CALLS_AT')

    return qa


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT_CORPUS, exist_ok=True)
    os.makedirs(OUT_BENCH, exist_ok=True)

    articles = build_articles(rng)
    df = pd.DataFrame(articles)
    df['relations'] = df['relations'].apply(json.dumps, ensure_ascii=False) \
        if False else df['relations'].apply(lambda r: json.dumps(r, ensure_ascii=False))
    df.to_csv(os.path.join(OUT_CORPUS, 'maritime_corpus.csv'), index=False)

    entities = {
        'companies': COMPANIES,
        'vessels': [{'name': v, 'type': t, 'operator': c} for v, t, c in VESSELS],
        'ports': PORTS,
        'calls_at': CALLS_AT,
        'regulations': [{'name': r, 'applies_to': t, 'description': d} for r, t, d in REGULATIONS],
        'incidents': [{'id': i, 'vessel': v, 'port': p, 'description': d} for i, v, p, d in INCIDENTS],
    }
    with open(os.path.join(OUT_CORPUS, 'entities.json'), 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)

    qa = build_benchmark()
    with open(os.path.join(OUT_BENCH, 'qa_benchmark.json'), 'w', encoding='utf-8') as f:
        json.dump(qa, f, ensure_ascii=False, indent=2)

    n_hops = pd.Series([q['hops'] for q in qa]).value_counts().to_dict()
    print(f'articles: {len(articles)} (noise: {sum(1 for a in articles if a["relations"] == "[]" or not a["relations"])})')
    print(f'QA pairs: {len(qa)} by hops: {n_hops}')
    print('saved: maritime_corpus.csv, entities.json, qa_benchmark.json')


if __name__ == '__main__':
    main()
