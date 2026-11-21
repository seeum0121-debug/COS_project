"""산출물 JSON/CSV에서 결과보고서.html 생성.

재실행만 하면 최신 결과가 반영된다. in_domain_summary.json이 아직 없으면
해당 절은 '실행 중'으로 표시한다.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    p = HERE / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(v, nd=3, sign=False):
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if f != f:  # NaN
        return "—"
    return f"{f:+.{nd}f}" if sign else f"{f:.{nd}f}"


def ci(lo, hi):
    if lo is None or hi is None or lo != lo or hi != hi:
        return "—"
    return f"[{float(lo):+.3f}, {float(hi):+.3f}]"


def verdict(rho, lo, hi):
    """CI가 0을 제외하는지로 판정."""
    if lo is None or hi is None or lo != lo or hi != hi:
        return ("—", "neutral")
    if float(lo) * float(hi) > 0:
        return ("역방향" if float(rho) < 0 else "정방향", "bad" if float(rho) < 0 else "good")
    return ("신호 없음", "neutral")


def main() -> None:
    summary = load("summary.json") or {}
    robust = load("robustness.json") or {}
    ad = load("applicability_domain.json") or {}
    indom = load("in_domain_summary.json")
    ct = ad.get("component_tanimoto", {})
    fd = ad.get("feature_distance", {})

    rows_main = []
    for key, label in [("Tier1_single", "Tier 1 · 단일물질"),
                       ("Tier2_mixture", "Tier 2 · 이성분 혼합물")]:
        r = robust.get(key, {})
        cl = r.get("cluster_bootstrap_ci", [None, None])
        v, cls = verdict(r.get("rho_observed"), cl[0], cl[1])
        rows_main.append(f"""
        <tr>
          <td class="lbl">{label}</td>
          <td class="num">{r.get('n','—')}</td>
          <td class="num">{r.get('n_clusters','—')}</td>
          <td class="num strong">{fmt(r.get('rho_observed'), 3, True)}</td>
          <td class="num">{ci(cl[0], cl[1])}</td>
          <td class="num">{fmt(r.get('cluster_permutation_p'), 4)}</td>
          <td class="num">{fmt(r.get('loo_rho_min'),3,True)} ~ {fmt(r.get('loo_rho_max'),3,True)}</td>
          <td><span class="tag {cls}">{v}</span></td>
        </tr>""")

    # AD 임계값별
    if indom:
        ad_rows = []
        for r in indom.get("ad_results", []):
            if r["tier"] == "all":
                continue
            v, cls = verdict(r.get("rho"), r.get("ci_low"), r.get("ci_high"))
            tier = "단일" if r["tier"] == "Tier1_single" else "혼합"
            ad_rows.append(f"""
            <tr>
              <td class="lbl">{html.escape(r['ad_cut'])}</td>
              <td>{tier}</td>
              <td class="num">{r['n']}</td>
              <td class="num">{r.get('n_clusters','—')}</td>
              <td class="num strong">{fmt(r.get('rho'),3,True)}</td>
              <td class="num">{ci(r.get('ci_low'), r.get('ci_high'))}</td>
              <td class="num">{fmt(r.get('p'),4)}</td>
              <td><span class="tag {cls}">{v}</span></td>
            </tr>""")
        grad_rows = []
        for r in indom.get("distance_gradient", []):
            v, cls = verdict(r.get("rho"), r.get("ci_low"), r.get("ci_high"))
            grad_rows.append(f"""
            <tr>
              <td class="lbl">{html.escape(r['distance_tertile'])}</td>
              <td class="num">{html.escape(r['distance_range'])}</td>
              <td class="num">{r['n']}</td>
              <td class="num strong">{fmt(r.get('rho'),3,True)}</td>
              <td class="num">{ci(r.get('ci_low'), r.get('ci_high'))}</td>
              <td><span class="tag {cls}">{v}</span></td>
            </tr>""")
        sls_rows = []
        for r in indom.get("sls_sensitivity", []):
            tier = "단일" if r["tier"] == "Tier1_single" else "혼합"
            sls_rows.append(f"""
            <tr><td class="lbl">{html.escape(r['set'])}</td><td>{tier}</td>
            <td class="num">{r['n']}</td>
            <td class="num strong">{fmt(r.get('rho'),3,True)}</td>
            <td class="num">{ci(r.get('ci_low'), r.get('ci_high'))}</td>
            <td class="num">{fmt(r.get('p'),4)}</td></tr>""")
        g = {r["distance_tertile"]: r for r in indom.get("distance_gradient", [])}
        near, mid, far = g.get("가까움", {}), g.get("중간", {}), g.get("멂", {})
        adr = {(r["ad_cut"], r["tier"]): r for r in indom.get("ad_results", [])}
        t2_75 = adr.get(("p75", "Tier2_mixture"), {})
        t1_75 = adr.get(("p75", "Tier1_single"), {})
        indom_html = f"""
      <div class="callout red">
        <strong>핵심: Tier 2의 역상관은 적용범위 안에서 사라집니다.</strong>
        전체에서 rho {fmt(robust.get('Tier2_mixture',{}).get('rho_observed'),3,True)}였던 혼합물 상관이
        p75 이내에서는 {fmt(t2_75.get('rho'),3,True)} (CI {ci(t2_75.get('ci_low'), t2_75.get('ci_high'))}),
        p95 이내에서는 {fmt(adr.get(('p95','Tier2_mixture'),{}).get('rho'),3,True)}로 0 근처입니다.
        따라서 <b>모델이 화장품 계면활성제에 대해 체계적으로 뒤집혀 있다는 해석은 지지되지 않습니다.</b>
        앞서 보고한 역상관은 학습 영역 밖에서 발생한 현상입니다.
      </div>
      <div class="callout">
        <strong>다만 세 가지를 과장하면 안 됩니다.</strong>
        <b>(1) 거리 기울기가 단조롭지 않습니다.</b> 가장 강한 음의 상관은 가장 먼 구간이 아니라
        <b>중간 구간</b>({fmt(mid.get('rho'),3,True)}, CI {ci(mid.get('ci_low'), mid.get('ci_high'))})에
        몰려 있고, 가장 먼 구간은 오히려 양수({fmt(far.get('rho'),3,True)})입니다.
        순수한 '거리가 멀수록 악화' 설명으로는 전부 설명되지 않으며, 중간 구간의 화학적 구성을
        따로 확인해야 합니다.
        <b>(2) AD 내부의 Tier 1 양의 상관은 통계적으로 뒷받침되지 않습니다.</b>
        p75에서 rho {fmt(t1_75.get('rho'),3,True)}이지만 CI {ci(t1_75.get('ci_low'), t1_75.get('ci_high'))},
        p={fmt(t1_75.get('p'),4)}, 군집 {t1_75.get('n_clusters','—')}개뿐입니다. 경향일 뿐 증거가 아닙니다.
        <b>(3) p50 이내 표본은 0건입니다.</b> 외부 배합의 최소 거리
        ({fmt(fd.get('external_min'),3)})가 학습 중앙값({fmt(fd.get('training_median'),3)})을 이미 넘습니다.
        즉 '적용범위 내부'라 부른 구간조차 학습 분포의 가장자리입니다.
      </div>
      <table>
        <thead><tr><th>AD 기준</th><th>티어</th><th>n</th><th>군집</th>
        <th>rho</th><th>군집 부트스트랩 95% CI</th><th>치환 p</th><th>판정</th></tr></thead>
        <tbody>{''.join(ad_rows)}</tbody>
      </table>
      <h3>혼합물 · 학습공간 거리 3분위별 기울기</h3>
      <p class="note">외삽 가설의 직접 검정. 가까운 구간에서 rho가 0으로 수렴하면 외삽 설명이
      지지되고, 가까운 구간에서도 음수면 모델 방향 자체의 문제다.</p>
      <table>
        <thead><tr><th>구간</th><th>거리 범위</th><th>n</th><th>rho</th>
        <th>95% CI</th><th>판정</th></tr></thead>
        <tbody>{''.join(grad_rows)}</tbody>
      </table>
      <h3>학습 중복(SLS) 제외 민감도</h3>
      <table>
        <thead><tr><th>표본</th><th>티어</th><th>n</th><th>rho</th>
        <th>95% CI</th><th>치환 p</th></tr></thead>
        <tbody>{''.join(sls_rows)}</tbody>
      </table>"""
    else:
        indom_html = ('<p class="pending">실행 중입니다. '
                      '<code>python in_domain_analysis.py</code> 완료 후 '
                      '<code>python build_report.py</code>를 다시 실행하면 이 절이 채워집니다.</p>')

    fd = ad.get("feature_distance", {})
    res_rows = []
    for r in summary.get("results", []):
        if r["tier"] == "all":
            continue
        tier = "단일" if r["tier"] == "Tier1_single" else "혼합"
        res_rows.append(f"""<tr><td class="lbl">{r['concentration_pct']}</td><td>{tier}</td>
        <td class="num">{r['n']}</td><td class="num">{fmt(r.get('spearman'),3,True)}</td>
        <td class="num">{fmt(r.get('p_value'),4)}</td></tr>""")

    doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>화장품 계면활성제 외부 비교</title>
<style>
:root {{
  --bg:#ffffff; --fg:#17181c; --muted:#5d6370; --line:#e3e6ec; --soft:#f6f7f9;
  --accent:#2f5fd0; --bad:#b4342b; --bad-bg:#fdf2f1; --good:#1d7a4c; --good-bg:#f0f9f4;
  --warn:#8a5a00; --warn-bg:#fdf6e6;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#15171b; --fg:#e9ecf1; --muted:#9aa3b2; --line:#2b2f37; --soft:#1c1f25;
    --accent:#7da4f5; --bad:#f08279; --bad-bg:#2b1b1a; --good:#6fd39c; --good-bg:#15241c;
    --warn:#e2b55c; --warn-bg:#2a2317;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#15171b; --fg:#e9ecf1; --muted:#9aa3b2; --line:#2b2f37; --soft:#1c1f25;
  --accent:#7da4f5; --bad:#f08279; --bad-bg:#2b1b1a; --good:#6fd39c; --good-bg:#15241c;
  --warn:#e2b55c; --warn-bg:#2a2317;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font-family:-apple-system,"Segoe UI","Malgun Gothic",sans-serif;
  line-height:1.65; font-size:15px; }}
.wrap {{ max-width:960px; margin:0 auto; padding:48px 16px 80px; }}
h1 {{ font-size:1.65rem; margin:0 0 6px; letter-spacing:-0.01em; }}
h2 {{ font-size:1.18rem; margin:44px 0 12px; padding-bottom:7px;
  border-bottom:1px solid var(--line); }}
h3 {{ font-size:1rem; margin:28px 0 8px; color:var(--muted);
  text-transform:none; letter-spacing:0; }}
.sub {{ color:var(--muted); font-size:.9rem; margin:0 0 28px; }}
.callout {{ border-left:3px solid var(--warn); background:var(--warn-bg);
  padding:13px 16px; border-radius:0 6px 6px 0; margin:18px 0; font-size:.93rem; }}
.callout.red {{ border-color:var(--bad); background:var(--bad-bg); }}
.callout strong {{ display:block; margin-bottom:4px; }}
table {{ width:100%; border-collapse:collapse; margin:14px 0; font-size:.88rem; }}
th, td {{ padding:9px 10px; text-align:left; border-bottom:1px solid var(--line); }}
th {{ background:var(--soft); font-weight:600; font-size:.8rem; color:var(--muted);
  white-space:nowrap; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums;
  font-family:ui-monospace,"Cascadia Mono",Consolas,monospace; white-space:nowrap; }}
td.lbl {{ font-weight:600; }}
td.strong {{ font-weight:700; }}
.tag {{ display:inline-block; padding:2px 9px; border-radius:11px; font-size:.76rem;
  font-weight:600; white-space:nowrap; }}
.tag.bad {{ background:var(--bad-bg); color:var(--bad); }}
.tag.good {{ background:var(--good-bg); color:var(--good); }}
.tag.neutral {{ background:var(--soft); color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:11px; margin:16px 0; }}
.stat {{ border:1px solid var(--line); border-radius:8px; padding:13px 15px;
  background:var(--soft); }}
.stat .k {{ font-size:.76rem; color:var(--muted); margin-bottom:3px; }}
.stat .v {{ font-size:1.3rem; font-weight:700;
  font-variant-numeric:tabular-nums; letter-spacing:-0.02em; }}
.stat .u {{ font-size:.76rem; color:var(--muted); font-weight:400; }}
.note {{ color:var(--muted); font-size:.87rem; }}
.pending {{ color:var(--muted); background:var(--soft); padding:14px 16px;
  border-radius:7px; font-size:.9rem; }}
code {{ font-family:ui-monospace,Consolas,monospace; font-size:.85em;
  background:var(--soft); padding:1.5px 5px; border-radius:4px; }}
ul {{ padding-left:20px; }} li {{ margin:5px 0; }}
footer {{ margin-top:56px; padding-top:18px; border-top:1px solid var(--line);
  color:var(--muted); font-size:.82rem; }}
</style></head><body><div class="wrap">

<h1>Eye A 모델 — 화장품 계면활성제 외부 비교</h1>
<p class="sub">Lechuga et al. 2025, ACS Omega 10(46):55868–55878 · 생성 {summary.get('generated','2026-11-09')}</p>

<div class="callout red">
  <strong>결론 먼저: 음성 결과입니다.</strong>
  전체 표본에서 단일 계면활성제는 신호가 없었고(rho {fmt(robust.get('Tier1_single',{}).get('rho_observed'),3,True)}),
  이성분 혼합물은 <b>역방향</b> 연관을 보였습니다
  (rho {fmt(robust.get('Tier2_mixture',{}).get('rho_observed'),3,True)},
  군집 치환검정 p={fmt(robust.get('Tier2_mixture',{}).get('cluster_permutation_p'),4)}).
  <b>단, 이 역상관은 적용범위 안에서 사라집니다</b>(아래 재분석 참조). 모델이 체계적으로
  뒤집혀 있다는 해석은 지지되지 않으며, 동시에 전이가 입증된 것도 아닙니다.
  모델·임계값·학습자료는 수정하지 않았고, 라벨을 예측에 맞추지 않았습니다.
</div>

<div class="callout">
  <strong>엔드포인트 주의</strong>
  zein number는 in vitro 단백질 변성 지표로 <b>EPA 눈자극 등급이 아닙니다</b>.
  따라서 이것은 EPA 분류의 외부검증이 아니라 모델 점수와 독립 실험값의
  순위 일치도 비교입니다. 민감도·특이도는 1차 결과로 보고하지 않습니다.
</div>

<h2>자료</h2>
<div class="grid">
  <div class="stat"><div class="k">원자료</div><div class="v">{summary.get('n_raw_rows','—')}<span class="u"> 행</span></div></div>
  <div class="stat"><div class="k">분석단위</div><div class="v">{summary.get('n_analysis_units','—')}<span class="u"> 행</span></div></div>
  <div class="stat"><div class="k">고유 배합</div><div class="v">{summary.get('n_unique_formulations','—')}<span class="u"> 종</span></div></div>
  <div class="stat"><div class="k">계면활성제</div><div class="v">{summary.get('n_surfactants','—')}<span class="u"> 종</span></div></div>
</div>
<p class="note">농도 0.5 / 1.0 / 2.0% 3수준. 모델에 농도 피처가 없으므로(<code>concentration_features=false</code>)
농도를 교란으로 처리해 층별 분석했고, 동일 (배합, 농도)의 반복측정은 평균 집계했습니다.
Tier 1은 단일물질로 분자 표현 자체의 신호를, Tier 2는 이성분 혼합물로 가중평균 집계 규칙을 봅니다.</p>

<h2>주 결과 — 군집 보정 Spearman</h2>
<table>
  <thead><tr><th>티어</th><th>n</th><th>군집</th><th>rho</th>
  <th>군집 부트스트랩 95% CI</th><th>치환 p</th><th>1종 제외 범위</th><th>판정</th></tr></thead>
  <tbody>{''.join(rows_main)}</tbody>
</table>
<div class="callout">
  <strong>비독립성 보정</strong>
  혼합물 분석단위는 22종 계면활성제의 조합이라 성분을 공유합니다. 행 단위
  Spearman은 p={fmt(next((r['p_value'] for r in summary.get('results',[]) if r['concentration_pct']=='pooled' and r['tier']=='Tier2_mixture'), None),6)}로
  나오지만 이는 과장된 값입니다. 배합 단위 치환검정 결과인
  p={fmt(robust.get('Tier2_mixture',{}).get('cluster_permutation_p'),4)}를 보고값으로 씁니다.
</div>

<h3>농도층별 (참고)</h3>
<table>
  <thead><tr><th>농도 %</th><th>티어</th><th>n</th><th>rho</th><th>p (행단위)</th></tr></thead>
  <tbody>{''.join(res_rows)}</tbody>
</table>

<h2>적용범위</h2>
<div class="grid">
  <div class="stat"><div class="k">Tanimoto 중앙값</div><div class="v">{fmt(ct.get('median'),3)}</div></div>
  <div class="stat"><div class="k">Tanimoto &lt; 0.4</div><div class="v">{ct.get('n_below_0.4','—')}<span class="u"> / 22</span></div></div>
  <div class="stat"><div class="k">학습 95분위 밖</div><div class="v">{fmt(fd.get('pct_external_beyond_training_p95'),1)}<span class="u"> %</span></div></div>
  <div class="stat"><div class="k">외부 / 학습 중앙거리</div><div class="v">{fmt(fd.get('external_median'),2)}<span class="u"> / {fmt(fd.get('training_median'),2)}</span></div></div>
</div>
<p class="note">외부 배합의 약 절반이 학습 화학공간 밖입니다. 글루코사이드·글리세레스
계열이 가장 멀고(Tanimoto 0.29–0.33), 설페이트 계열이 가장 가깝습니다.
따라서 Tier 2의 역상관은 "모델이 틀렸다"보다 <b>"학습 영역 밖에서 가중평균 집계가
순서를 보존하지 못한다"</b>로 읽는 것이 자료에 부합합니다. 다만 이 자료만으로
두 해석을 완전히 분리할 수는 없습니다.</p>

<h2>적용범위 내부 재분석</h2>
<p class="note">사후 부분집합 분석입니다. 임계값을 하나만 고르면 체리피킹이 되므로
관례적 분위수(p50·p75·p90·p95)를 전부 보고합니다. 표본이 줄어 검정력이 낮으므로
유의성보다 CI 폭과 추정치 안정성을 봅니다.</p>
{indom_html}

<h2>한계</h2>
<ul>
  <li><b>엔드포인트 불일치.</b> zein number ≠ EPA 눈자극 등급. 순위 비교만 유효합니다.</li>
  <li><b>대표구조 근사.</b> 22종 모두 다분산 상업 계면활성제로 단일 SMILES가 없습니다.
      사전 지정 규칙 R1–R9로 구성했고 결과를 보고 조정하지 않았습니다.
      coco 계열을 C12로 고정해 Coco Glucoside와 Lauryl Glucoside의 대표구조가
      동일해졌고, laureth sulfate·coceth sulfate의 EO 수는 각각 2, 3으로 가정했습니다.</li>
  <li><b>학습 중복 1종.</b> Sodium lauryl sulfate가 학습 배합 1건과 구조 동일(Tanimoto 1.000).
      22종 중 1종이라 영향은 제한적이며 제외 민감도를 함께 보고합니다.</li>
  <li><b>검정력.</b> 고유 배합 66종, 군집 44개로 CI가 넓습니다.</li>
  <li><b>in vitro 단일 지표.</b> zein number 한 종류만으로 자극성 전체를 대표하지 않습니다.</li>
</ul>

<h2>재현</h2>
<ul>
  <li><code>python resolve_structures.py</code> — 22종 대표구조 생성</li>
  <li><code>python run_validation.py</code> — 피처·예측·순위상관</li>
  <li><code>python robustness.py</code> — 군집 치환검정·군집 부트스트랩·영향도</li>
  <li><code>python applicability_domain.py</code> — Tanimoto 및 피처공간 거리</li>
  <li><code>python in_domain_analysis.py</code> — AD 내부 재분석·거리 기울기·SLS 민감도</li>
  <li><code>python build_report.py</code> — 본 보고서 재생성</li>
</ul>

<footer>
  모델 identity <code>{html.escape(str(summary.get('model_identity','—'))[:16])}…</code> ·
  임계값 {fmt(summary.get('threshold'),4)} ·
  부트스트랩 {summary.get('bootstrap_iterations','—')}회 · seed {summary.get('seed','—')}<br>
  출처 SHA-256 <code>{html.escape(str(summary.get('source_sha256','—'))[:16])}…</code> ·
  모델 SHA-256 <code>{html.escape(str(summary.get('model_sha256','—'))[:16])}…</code>
</footer>

</div></body></html>"""

    out = HERE / "결과보고서.html"
    out.write_text(doc, encoding="utf-8")
    print(f"저장: {out}")
    print(f"AD 재분석 절: {'반영됨' if indom else '대기 중 (in_domain_analysis.py 실행 후 재생성)'}")


if __name__ == "__main__":
    main()
