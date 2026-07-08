"""
Cross-document insights from the KMST accident graph.

Every verdict document describes ONE accident. None of them can answer:
  Q1  Which cause categories co-occur most often, per accident type?
  Q2  Do night collisions have a different leading-cause profile than daytime?
  Q3  Which cause chains (A -> B) repeat across accidents?
  Q4  How does the cause profile differ between fishing vessels and merchant ships?
  Q5  Which cause categories draw the heaviest sanctions (license-suspension months)?

These are joins ACROSS accidents through the shared CauseCategory taxonomy —
the knowledge the graph creates that no single document contains.

Usage: python evaluation/accident_insights.py
Saves docs/analysis/accident_insights.png + prints findings.
"""
import os
from collections import Counter

import neo4j
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

load_dotenv()
URI = os.getenv('NEO4J_URI', 'neo4j://127.0.0.1:7687')
AUTH = ('neo4j', os.getenv('NEO4J_PASSWORD', '12345678'))
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# Korean font (macOS/Linux fallbacks)
for cand in ['AppleGothic', 'NanumGothic', 'Malgun Gothic']:
    if any(cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = cand
        break
plt.rcParams['axes.unicode_minus'] = False

driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()


def q(cypher, **params):
    with driver.session() as s:
        return s.run(cypher, **params).data()


n_acc = q("MATCH (a:Accident) RETURN count(*) AS n")[0]['n']
print(f'=== KMST accident graph: {n_acc} adjudicated accidents ===\n')

# ---- Q1: cause-category frequency by accident type ----
rows = q("""
    MATCH (a:Accident)-[:HAS_CAUSE]->(:Cause)-[:OF_TYPE]->(cat:CauseCategory)
    RETURN a.type AS atype, cat.name AS cat, count(*) AS n
    ORDER BY n DESC
""")
print('Q1. 사고유형 × 원인 카테고리 (상위 12):')
for r in rows[:12]:
    print(f"   {r['atype']:<8} × {r['cat']:<14} {r['n']}")

# ---- Q2: night vs day cause profile for collisions ----
nd = q("""
    MATCH (a:Accident)-[:HAS_CAUSE]->(:Cause)-[:OF_TYPE]->(cat:CauseCategory)
    WHERE a.type IN ['충돌', '접촉'] AND a.night IS NOT NULL
    RETURN a.night AS night, cat.name AS cat, count(*) AS n
""")
print('\nQ2. 충돌·접촉 사고: 야간 vs 주간 원인 분포:')
night_c = Counter({r['cat']: r['n'] for r in nd if r['night']})
day_c = Counter({r['cat']: r['n'] for r in nd if not r['night']})
for cat in (night_c + day_c).most_common(8):
    print(f"   {cat[0]:<16} 야간 {night_c.get(cat[0], 0):>2}  주간 {day_c.get(cat[0], 0):>2}")

# ---- Q3: repeated cause chains (category level) ----
chains = q("""
    MATCH (a:Accident)-[:HAS_CAUSE]->(c1:Cause)-[:LEADS_TO]->(c2:Cause)
    MATCH (c1)-[:OF_TYPE]->(k1:CauseCategory), (c2)-[:OF_TYPE]->(k2:CauseCategory)
    RETURN k1.name + ' → ' + k2.name AS chain, count(DISTINCT a) AS n
    ORDER BY n DESC LIMIT 10
""")
print('\nQ3. 반복되는 원인 사슬 (카테고리 수준, 사고 수 기준):')
for r in chains:
    print(f"   [{r['n']:>2}건] {r['chain']}")

# ---- Q4: fishing vs merchant cause profile ----
fv = q("""
    MATCH (a:Accident)-[:INVOLVES]->(v:AVessel),
          (a)-[:HAS_CAUSE]->(:Cause)-[:OF_TYPE]->(cat:CauseCategory)
    WITH a, cat, collect(DISTINCT v.type) AS vtypes
    RETURN CASE WHEN '어선' IN vtypes THEN '어선 관련' ELSE '비어선' END AS grp,
           cat.name AS cat, count(DISTINCT a) AS n
""")
print('\nQ4. 어선 관련 vs 비어선 사고의 원인 프로필:')
fish = Counter({}); nonf = Counter({})
for r in fv:
    (fish if r['grp'] == '어선 관련' else nonf)[r['cat']] += r['n']
for cat, _ in (fish + nonf).most_common(8):
    print(f"   {cat:<16} 어선 {fish.get(cat, 0):>2}  비어선 {nonf.get(cat, 0):>2}")

# ---- Q5: sanction severity by cause category ----
sanc = q("""
    MATCH (a:Accident)-[:HAS_CAUSE]->(:Cause)-[:OF_TYPE]->(cat:CauseCategory),
          (a)-[:IMPOSED]->(s:Sanction)
    WHERE s.months IS NOT NULL
    RETURN cat.name AS cat, avg(s.months) AS avg_m, count(DISTINCT a) AS n
    ORDER BY avg_m DESC LIMIT 10
""")
print('\nQ5. 원인 카테고리별 평균 업무정지 개월 (면허 처분 기준):')
for r in sanc:
    print(f"   {r['cat']:<16} 평균 {r['avg_m']:.1f}개월 (사고 {r['n']}건)")

# ---------------- figure ----------------
fig, axs = plt.subplots(1, 3, figsize=(19, 6))

# (1) accident type x cause category heatmap
atypes = sorted({r['atype'] for r in rows if r['atype']})
cats = [c for c, _ in Counter({r['cat']: r['n'] for r in rows}).most_common(10)]
M = np.zeros((len(cats), len(atypes)))
for r in rows:
    if r['cat'] in cats and r['atype'] in atypes:
        M[cats.index(r['cat']), atypes.index(r['atype'])] = r['n']
im = axs[0].imshow(M, cmap='Blues', aspect='auto')
axs[0].set_xticks(range(len(atypes)), atypes, rotation=30, fontsize=9, ha='right')
axs[0].set_yticks(range(len(cats)), cats, fontsize=9)
for i in range(len(cats)):
    for j in range(len(atypes)):
        if M[i, j] > 0:
            axs[0].text(j, i, int(M[i, j]), ha='center', va='center', fontsize=8,
                        color='white' if M[i, j] > M.max() * 0.6 else '#1E293B')
axs[0].set_title(f'사고유형 × 원인 카테고리 ({n_acc}건 재결서)', fontsize=12)

# (2) repeated cause chains
if chains:
    labels = [r['chain'] for r in chains][::-1]
    vals = [r['n'] for r in chains][::-1]
    axs[1].barh(labels, vals, color='#0F4C81')
    axs[1].set_title('반복되는 원인 사슬 (선행 → 직접 원인)', fontsize=12)
    axs[1].set_xlabel('사고 수')
    axs[1].tick_params(axis='y', labelsize=8.5)
    axs[1].grid(alpha=0.3, axis='x')

# (3) fishing vs non-fishing profile
top_cats = [c for c, _ in (fish + nonf).most_common(8)]
x = np.arange(len(top_cats))
axs[2].bar(x - 0.2, [fish.get(c, 0) for c in top_cats], 0.4, color='#2E6F9E', label='어선 관련')
axs[2].bar(x + 0.2, [nonf.get(c, 0) for c in top_cats], 0.4, color='#C4762E', label='비어선')
axs[2].set_xticks(x, top_cats, rotation=30, fontsize=8.5, ha='right')
axs[2].set_title('어선 관련 여부에 따른 원인 프로필', fontsize=12)
axs[2].set_ylabel('사고 수')
axs[2].legend()
axs[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
out = os.path.join(OUT_DIR, 'accident_insights.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f'\nsaved: {out}')
print('INSIGHTS_OK')
