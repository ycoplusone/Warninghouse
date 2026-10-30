"""오늘의 시장 상태 — 한 장 요약 HTML 리포트 (reports/market_today.html)

③ indicators 결과(data/signals, data/features)를 읽어, 숫자보다 문장 위주로 오늘의 상태를 정리한다.
브라우저로 열어 본다. 날짜별 사본은 reports/daily/market_<기준일>.html에 남긴다.

담는 내용
  1. 오늘의 경보 단계 (정상/주의/위험)와 그 의미
  2. 왜 그렇게 판단했나: 카나리아 자산·VIX 각각의 상태 + 최근 1년 흐름
  3. 최근 1년 경보 기록 (날짜별 색 띠)
  4. 전략이라면 지금 어떤 비중일지 (미국·한국)
  5. 관심 종목 한눈에 (추세, 과열 여부, 1개월 수익률)
  6. 이 신호를 얼마나 믿어도 되나 (백테스트 결과)
"""

import html
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import risk, storage
from src.collect import is_korean_ticker
from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# 해외 티커의 쉬운 이름 (국내 종목은 파일 이름의 한글 종목명을 쓴다)
FRIENDLY_NAMES = {
    "SPY": "미국 S&P500",
    "QQQ": "미국 나스닥100",
    "SHY": "미국 단기국채",
    "IEF": "미국 중기국채",
    "BIL": "미국 초단기국채",
    "GLD": "금",
    "UUP": "달러",
    "EEM": "신흥국 주식",
    "HYG": "하이일드 채권",
    "^VIX": "VIX 공포지수",
}

CANARY_EXPLAIN = {
    "EEM": "세계 투자자들이 위험을 피하기 시작하면 가장 먼저 팔리는 경향이 있습니다.",
    "HYG": "신용도가 낮은 회사들의 채권입니다. 기업 부도 걱정이 커지면 먼저 떨어집니다.",
}

METHOD_EXPLAIN = {
    "momentum": "최근 1·3·6·12개월 수익률을 섞은 추세 점수",
    "sma200": "200일 평균선 대비 위치",
}

# 경보 단계별 표시 (상태색은 항상 아이콘 + 글자와 함께 쓴다)
LEVEL_STYLE = {
    0: {"label": "정상", "icon": "✓", "var": "--status-good"},
    1: {"label": "주의", "icon": "!", "var": "--status-warning"},
    2: {"label": "위험", "icon": "✕", "var": "--status-critical"},
}

VIX_CALM = 20  # 이 아래면 '안정'으로 표시 (표시용 기준)
LOW_VOL = 0.02  # 최근 연 변동성이 2% 미만이면 현금성 자산으로 보고 RSI 해석을 생략


def friendly_name(ticker: str, path: Path | None) -> str:
    if ticker in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[ticker]
    name = storage.name_from_path(path, ticker) if path else None
    return name.replace("_", " ") if name else ticker


def esc(text) -> str:
    return html.escape(str(text))


# ---------------------------------------------------------------------------
# 작은 그래프 (SVG)
# ---------------------------------------------------------------------------

def dot_svg(cx: float, cy: float) -> str:
    """바탕색 테두리가 있는 원형 점 (테두리 원 위에 색 원)."""
    return (f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" class="dot-ring" />'
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" class="dot-fill" />')


def sparkline_svg(series: pd.Series, zero_line: float | None, value_fmt: str, label: str) -> str:
    """최근 1년 흐름 작은 선 그래프. 마우스를 올리면 날짜·값이 보인다."""
    s = series.dropna()
    w, h, pad = 300, 70, 6
    lo = min(s.min(), zero_line if zero_line is not None else s.min())
    hi = max(s.max(), zero_line if zero_line is not None else s.max())
    span = (hi - lo) or 1

    def x(i):
        return pad + i * (w - 2 * pad) / max(len(s) - 1, 1)

    def y(v):
        return h - pad - (v - lo) / span * (h - 2 * pad)

    points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(s.values))
    zero = ""
    if zero_line is not None:
        zero = (f'<line x1="{pad}" x2="{w - pad}" y1="{y(zero_line):.1f}" y2="{y(zero_line):.1f}" '
                f'class="zero" />')
    data = json.dumps({
        "dates": [d.strftime("%Y-%m-%d") for d in s.index],
        "values": [round(float(v), 4) for v in s.values],
        "fmt": value_fmt, "label": label,
        "xs": [round(x(i), 1) for i in range(len(s))], "ys": [round(y(v), 1) for v in s.values],
    })
    return (
        f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img" '
        f'aria-label="{esc(label)} 최근 1년 흐름" data-spark=\'{esc(data)}\'>'
        f'{zero}<polyline points="{points}" class="line" />'
        # 점은 길이 0인 선 + 둥근 끝으로 그린다. 그래프가 가로로 늘어나도 원 모양이 유지된다.
        f'<g class="end-dot">{dot_svg(x(len(s) - 1), y(s.iloc[-1]))}</g>'
        f'<line class="cross" x1="0" x2="0" y1="0" y2="{h}" visibility="hidden" />'
        f'<g class="hover-dot" visibility="hidden">{dot_svg(0, 0)}</g>'
        f'</svg>'
    )


def timeline_svg(level: pd.Series) -> str:
    """최근 1년 경보 단계 색 띠. 하루가 한 칸."""
    rects = []
    for i, v in enumerate(level.values):
        style = LEVEL_STYLE.get(int(v)) if pd.notna(v) else None
        fill = f"var({style['var']})" if style else "var(--border)"
        rects.append(f'<rect x="{i}" y="0" width="1.02" height="10" fill="{fill}" />')
    data = json.dumps({
        "dates": [d.strftime("%Y-%m-%d") for d in level.index],
        "labels": [LEVEL_STYLE[int(v)]["label"] if pd.notna(v) else "-" for v in level.values],
    })
    return (
        f'<svg class="timeline" viewBox="0 0 {len(level)} 10" preserveAspectRatio="none" role="img" '
        f'aria-label="최근 1년 경보 단계 기록" data-timeline=\'{esc(data)}\'>{"".join(rects)}</svg>'
    )


# ---------------------------------------------------------------------------
# 내용 만들기
# ---------------------------------------------------------------------------

def status_chip(level: int) -> str:
    st = LEVEL_STYLE[level]
    return (f'<span class="chip"><span class="dot" style="background:var({st["var"]})">{st["icon"]}</span>'
            f'{st["label"]}</span>')


def trend_chip(up: bool, up_text: str = "상승 추세", down_text: str = "하락 추세") -> str:
    return status_chip(0).replace("정상", up_text) if up else status_chip(2).replace("위험", down_text)


def level_sentence(level: int, bad: int, n: int, vix_alert: bool, equity: float) -> str:
    if level == 0:
        why = f"카나리아 {n}개가 모두 상승 추세이고 VIX도 안정적입니다."
    elif level == 1:
        why = f"카나리아 {n}개 중 {bad}개가 하락 추세로 돌아섰습니다."
    elif vix_alert:
        why = "VIX(공포지수)가 경보 기준을 넘었습니다."
    else:
        why = f"카나리아 {n}개가 모두 하락 추세입니다."
    return f"{why} 이 전략이라면 주식 비중을 <strong>{equity:.0%}</strong>로 둡니다."


def since_text(level: pd.Series) -> str:
    """현재 단계가 언제부터 이어졌는지."""
    lv = level.dropna()
    changed = lv.ne(lv.shift())
    start = lv.index[changed][-1]
    days = len(lv.loc[start:])
    prev = lv.loc[:start].iloc[-2] if len(lv.loc[:start]) > 1 else None
    prev_text = f" (그 전에는 '{LEVEL_STYLE[int(prev)]['label']}')" if prev is not None else ""
    return f"{start.date()}부터 {days}거래일째{prev_text}"


def allocation_rows(uni: dict, equity: float, names: dict[str, str]) -> list[tuple[str, float]]:
    """주식 비중 equity일 때 종목별 비중."""
    rows = [(names.get(t, t), equity / len(uni["offensive"])) for t in uni["offensive"]]
    rows += [(names.get(t, t), (1 - equity) / len(uni["defensive"])) for t in uni["defensive"]]
    return rows


def allocation_table(title: str, now_level: int, next_level: int, uni: dict, equity_by_level: dict, names: dict) -> str:
    now = allocation_rows(uni, equity_by_level[now_level], names)
    nxt = allocation_rows(uni, equity_by_level[next_level], names)
    body = "".join(
        f"<tr><td>{esc(n)}</td><td class='num'>{a:.0%}</td><td class='num'>{b:.0%}</td></tr>"
        for (n, a), (_, b) in zip(now, nxt)
    )
    return (
        f"<div class='alloc'><h3>{esc(title)}</h3><table>"
        f"<thead><tr><th>자산</th><th class='num'>지금 보유<br><small>지난달 말 {LEVEL_STYLE[now_level]['label']}</small></th>"
        f"<th class='num'>다음 달<br><small>오늘 {LEVEL_STYLE[next_level]['label']} 유지 시</small></th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def watchlist_rows(features_dir: Path, exclude: set[str]) -> tuple[list[str], list[str]]:
    """관심 종목 표의 행 (미국, 한국)."""
    us, kr = [], []
    for ticker in storage.list_tickers(features_dir):
        if ticker in exclude:
            continue
        path = storage.find_path(features_dir, ticker)
        df = storage.load_prices(path)
        last = df.iloc[-1]
        month = df["Close"].iloc[-1] / df["Close"].iloc[-22] - 1 if len(df) > 22 else float("nan")

        if pd.notna(last.get("Trend_Up")):
            trend = trend_chip(bool(last["Trend_Up"]))
        else:
            trend = "-"
        rsi = last.get("RSI")
        # 현금성 자산(초단기채 등)은 매일 조금씩만 올라 RSI가 100 근처로 붙는다 → 과열로 읽으면 안 됨
        vol = df["Return"].iloc[-60:].std() * (252 ** 0.5)
        if pd.isna(rsi):
            heat = "-"
        elif vol < LOW_VOL:
            heat = f"변동 거의 없음 <small>(현금성 자산)</small>"
        elif rsi >= 70:
            heat = f"많이 올라 과열 <small>(RSI {rsi:.0f})</small>"
        elif rsi <= 30:
            heat = f"많이 빠져 침체 <small>(RSI {rsi:.0f})</small>"
        else:
            heat = f"보통 <small>(RSI {rsi:.0f})</small>"
        month_cls = "up" if month > 0 else "down"
        month_text = f"<span class='{month_cls}'>{'▲' if month > 0 else '▼'} {abs(month):.1%}</span>"

        row = (f"<tr><td>{esc(friendly_name(ticker, path))}<br><small>{esc(ticker)} · {df.index[-1].date()}</small></td>"
               f"<td>{trend}</td><td>{heat}</td><td class='num'>{month_text}</td></tr>")
        (kr if is_korean_ticker(ticker) else us).append(row)
    return us, kr


def trust_box(report_dir: Path) -> str:
    """백테스트 결과로 '이 신호를 얼마나 믿을 수 있나'를 설명."""
    path = report_dir / "backtest_summary.csv"
    if not path.exists():
        return "<p>아직 백테스트 결과가 없습니다. <code>python main.py --step backtest</code>를 실행하면 여기에 요약이 나옵니다.</p>"
    summary = pd.read_csv(path, index_col=["시장", "전략"])
    items = []
    for market, label in [("US", "미국"), ("KR", "한국")]:
        if market not in summary.index.get_level_values(0):
            continue
        s = summary.loc[market]
        canary = s.loc["카나리아 단검"]
        base = s.loc["공격자산 보유(방어 없음)"]
        items.append(
            f"<li><strong>{label}</strong>: 이 전략은 그냥 들고 있는 것보다 "
            f"최악의 손실을 {abs(base['MDD']):.0%} → {abs(canary['MDD']):.0%}로 줄였지만, "
            f"연평균 수익률은 {base['CAGR']:.1%} → {canary['CAGR']:.1%}로 낮았습니다.</li>"
        )
    return (
        "<p>과거 데이터로 시험한 결과, 이 경보는 <strong>큰 위기를 일부 피했지만 너무 자주 울려서</strong> "
        "전체 수익은 오히려 줄었습니다.</p>"
        f"<ul>{''.join(items)}</ul>"
        "<p>그러니 '위험'이 떴다고 바로 팔기보다는, <strong>시장 분위기를 살피는 참고 자료</strong>로만 보세요.</p>"
    )


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------

STYLE = """
:root {
  color-scheme: light;
  --bg: #f4f3f0; --surface: #fcfcfb; --border: #e6e5e1;
  --text: #0b0b0b; --text-2: #52514e; --text-3: #75736d;
  --series-1: #2a78d6; --zero: #b9b7b0;
  --status-good: #0ca30c; --status-warning: #fab219; --status-critical: #d03b3b;
  --up: #b42318; --down: #1d5fb8;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #121211; --surface: #1a1a19; --border: #33322f;
    --text: #f0efec; --text-2: #c3c2b7; --text-3: #9a988f;
    --series-1: #3987e5; --zero: #5d5b55;
    --up: #ff7a6e; --down: #6ea8ff;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #121211; --surface: #1a1a19; --border: #33322f;
  --text: #f0efec; --text-2: #c3c2b7; --text-3: #9a988f;
  --series-1: #3987e5; --zero: #5d5b55;
  --up: #ff7a6e; --down: #6ea8ff;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", system-ui, sans-serif;
  line-height: 1.6; font-size: 15px;
}
main { max-width: 920px; margin: 0 auto; padding: 32px 16px 48px; }
header h1 { font-size: 26px; margin: 0; }
header p { margin: 4px 0 0; color: var(--text-2); font-size: 14px; }
section { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px 22px; margin-top: 16px; }
section h2 { font-size: 17px; margin: 0 0 12px; }
small { color: var(--text-3); font-size: 12px; }
.hero { display: flex; gap: 20px; align-items: center; }
.hero .big { width: 84px; height: 84px; border-radius: 50%; display: grid; place-items: center;
  font-size: 40px; font-weight: 700; color: #fff; flex: none; }
.hero .level { font-size: 30px; font-weight: 700; margin: 0; }
.hero p { margin: 4px 0 0; color: var(--text-2); }
.chip { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; white-space: nowrap; }
.chip .dot { width: 20px; height: 20px; border-radius: 50%; display: grid; place-items: center;
  color: #fff; font-size: 12px; font-weight: 700; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; }
.card { border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
.card h3 { margin: 0; font-size: 15px; }
.card .explain { color: var(--text-2); font-size: 13px; margin: 6px 0 8px; }
.card .value { font-size: 13px; color: var(--text-2); margin-top: 4px; }
.spark, .timeline { width: 100%; display: block; overflow: visible; }
.spark { height: 70px; margin-top: 4px; }
.spark .line { fill: none; stroke: var(--series-1); stroke-width: 2; vector-effect: non-scaling-stroke; }
.spark .zero { stroke: var(--zero); stroke-width: 1; stroke-dasharray: 4 3; vector-effect: non-scaling-stroke; }
.spark .dot-ring { stroke: var(--surface); stroke-width: 12; stroke-linecap: round; vector-effect: non-scaling-stroke; }
.spark .dot-fill { stroke: var(--series-1); stroke-width: 8; stroke-linecap: round; vector-effect: non-scaling-stroke; }
.spark .cross { stroke: var(--text-3); stroke-width: 1; vector-effect: non-scaling-stroke; }
.timeline { height: 28px; border-radius: 4px; }
.axis { display: flex; justify-content: space-between; color: var(--text-3); font-size: 12px; margin-top: 4px; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; font-size: 13px; color: var(--text-2); }
.allocs { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
.alloc h3 { font-size: 15px; margin: 0 0 6px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th { text-align: left; font-weight: 600; color: var(--text-2); font-size: 13px; border-bottom: 1px solid var(--border); padding: 6px 8px; vertical-align: bottom; }
td { border-bottom: 1px solid var(--border); padding: 8px; vertical-align: middle; }
.num { text-align: right; }
.up { color: var(--up); } .down { color: var(--down); }
.note { color: var(--text-2); font-size: 13px; margin: 10px 0 0; }
.table-wrap { overflow-x: auto; }
#tip { position: fixed; pointer-events: none; background: var(--text); color: var(--surface);
  padding: 4px 8px; border-radius: 6px; font-size: 12px; display: none; z-index: 10; white-space: nowrap; }
@media (max-width: 560px) {
  .hero { flex-direction: column; align-items: flex-start; }
  section { padding: 16px; }
}
"""

SCRIPT = """
const tip = document.getElementById('tip');
function showTip(text, e) { tip.textContent = text; tip.style.display = 'block';
  tip.style.left = Math.min(e.clientX + 12, window.innerWidth - tip.offsetWidth - 8) + 'px';
  tip.style.top = (e.clientY - 34) + 'px'; }
function hideTip() { tip.style.display = 'none'; }
function indexAt(svg, e, n) { const r = svg.getBoundingClientRect();
  return Math.max(0, Math.min(n - 1, Math.round((e.clientX - r.left) / r.width * (n - 1)))); }
document.querySelectorAll('svg[data-spark]').forEach(svg => {
  const d = JSON.parse(svg.dataset.spark);
  const cross = svg.querySelector('.cross'), dot = svg.querySelector('.hover-dot');
  svg.addEventListener('mousemove', e => {
    const i = indexAt(svg, e, d.values.length);
    cross.setAttribute('x1', d.xs[i]); cross.setAttribute('x2', d.xs[i]); cross.setAttribute('visibility', 'visible');
    dot.querySelectorAll('line').forEach(l => { l.setAttribute('x1', d.xs[i]); l.setAttribute('x2', d.xs[i]);
      l.setAttribute('y1', d.ys[i]); l.setAttribute('y2', d.ys[i]); });
    dot.setAttribute('visibility', 'visible');
    const v = d.values[i];
    const txt = d.fmt === 'pct' ? (v >= 0 ? '+' : '') + v.toFixed(2) : v.toFixed(1);
    showTip(d.dates[i] + '  ' + d.label + ' ' + txt, e);
  });
  svg.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); dot.setAttribute('visibility', 'hidden'); hideTip(); });
});
document.querySelectorAll('svg[data-timeline]').forEach(svg => {
  const d = JSON.parse(svg.dataset.timeline);
  svg.addEventListener('mousemove', e => { const i = indexAt(svg, e, d.dates.length); showTip(d.dates[i] + '  ' + d.labels[i], e); });
  svg.addEventListener('mouseleave', hideTip);
});
"""


def build_html(config: dict, signals: pd.DataFrame, features_dir: Path, report_dir: Path) -> tuple[str, pd.Timestamp]:
    canary_cfg = config["canary"]
    equity_by_level = {int(k): v for k, v in config["backtest"]["equity_by_level"].items()}
    sig = signals.dropna(subset=["Risk_Level"])
    last = sig.iloc[-1]
    as_of = sig.index[-1]
    level = int(last["Risk_Level"])
    n = len(canary_cfg["tickers"])
    year = sig.loc[as_of - pd.DateOffset(years=1):]

    # 지난달 말 신호 = 지금 보유 비중의 근거 (월말 리밸런싱)
    prev_month = sig.loc[: (as_of.to_period("M").start_time - pd.Timedelta(days=1))]
    held_level = int(prev_month["Risk_Level"].iloc[-1])

    # 1) 히어로
    st = LEVEL_STYLE[level]
    hero = (
        f"<section class='hero'><div class='big' style='background:var({st['var']})' aria-hidden='true'>{st['icon']}</div>"
        f"<div><p class='level'>경보 단계: {st['label']}</p>"
        f"<p>{level_sentence(level, int(last['Canary_Bad']), n, bool(last['VIX_Alert']), equity_by_level[level])}</p>"
        f"<p><small>{since_text(sig['Risk_Level'])}</small></p></div></section>"
    )

    # 2) 이유 카드
    cards = []
    method_text = METHOD_EXPLAIN.get(canary_cfg["method"], "추세 점수")
    for t in canary_cfg["tickers"]:
        score = sig[f"Score_{t}"]
        up = bool(score.iloc[-1] >= 0)
        cards.append(
            f"<div class='card'><h3>{esc(FRIENDLY_NAMES.get(t, t))} <small>{esc(t)}</small></h3>"
            f"<div>{trend_chip(up)}</div>"
            f"<p class='explain'>{esc(CANARY_EXPLAIN.get(t, ''))}</p>"
            f"{sparkline_svg(score.loc[year.index], 0.0, 'pct', '점수')}"
            f"<p class='value'>{esc(method_text)}: <strong>{score.iloc[-1]:+.2f}</strong> "
            f"<small>(점선 0보다 아래면 하락 추세)</small></p></div>"
        )
    vix_val, vix_thr = float(last["VIX"]), canary_cfg["vix_threshold"]
    if vix_thr is not None and vix_val > vix_thr:
        vix_chip, vix_text = status_chip(2).replace("위험", "경보"), f"기준값 {vix_thr}을 넘었습니다. 시장이 크게 겁먹은 상태입니다."
    elif vix_val >= VIX_CALM:
        vix_chip, vix_text = status_chip(1).replace("주의", "다소 높음"), "평소보다 불안하지만 경보 수준은 아닙니다."
    else:
        vix_chip, vix_text = status_chip(0).replace("정상", "안정"), "시장이 차분한 상태입니다."
    cards.append(
        f"<div class='card'><h3>VIX 공포지수 <small>^VIX</small></h3><div>{vix_chip}</div>"
        f"<p class='explain'>앞으로 주가가 얼마나 출렁일지 시장이 예상하는 값입니다. {esc(vix_text)}</p>"
        f"{sparkline_svg(year['VIX'], float(vix_thr) if vix_thr is not None else None, 'num', 'VIX')}"
        f"<p class='value'>현재 <strong>{vix_val:.1f}</strong> "
        f"<small>(점선 = 경보 기준 {vix_thr if vix_thr is not None else '없음'})</small></p></div>"
    )
    reasons = f"<section><h2>왜 이렇게 판단했나</h2><div class='cards'>{''.join(cards)}</div></section>"

    # 3) 1년 경보 기록
    share = year["Risk_Level"].value_counts(normalize=True)
    changes = int(year["Risk_Level"].ne(year["Risk_Level"].shift()).sum() - 1)
    legend = "".join(
        f"<span class='chip'><span class='dot' style='background:var({s['var']})'>{s['icon']}</span>"
        f"{s['label']} {share.get(k, 0):.0%}</span>"
        for k, s in LEVEL_STYLE.items()
    )
    mid = year.index[len(year) // 2]
    history = (
        f"<section><h2>최근 1년 경보 기록</h2>{timeline_svg(year['Risk_Level'])}"
        f"<div class='axis'><span>{year.index[0].date()}</span><span>{mid.date()}</span><span>{year.index[-1].date()}</span></div>"
        f"<div class='legend'>{legend}</div>"
        f"<p class='note'>지난 1년 동안 경보 단계가 <strong>{changes}번</strong> 바뀌었습니다. "
        f"자주 바뀔수록 신호가 예민하다는 뜻입니다. 이 전략은 매일이 아니라 <strong>매월 말</strong> 신호로만 비중을 바꿉니다.</p></section>"
    )

    # 4) 전략 비중
    names = {t: FRIENDLY_NAMES.get(t, t) for t in FRIENDLY_NAMES}
    for t in config.get("universe_kr", {}).get("offensive", []) + config.get("universe_kr", {}).get("defensive", []):
        names[t] = friendly_name(t, storage.find_path(features_dir, t))
    tables = [allocation_table("미국", held_level, level, config["universe"], equity_by_level, names)]
    if "universe_kr" in config:
        tables.append(allocation_table("한국", held_level, level, config["universe_kr"], equity_by_level, names))
    alloc = (
        f"<section><h2>이 전략이라면 지금</h2><div class='allocs'>{''.join(tables)}</div>"
        f"<p class='note'>이 전략은 매월 말 신호로만 비중을 바꿉니다. 그래서 '지금 보유'는 지난달 말 신호, "
        f"'다음 달'은 오늘 신호가 이달 말까지 이어질 때의 비중입니다. 교육용 예시이며, 한국은 미국 신호를 하루 늦게 반영하므로 실제 전환 시점이 하루 다를 수 있습니다.</p></section>"
    )

    # 5) 관심 종목
    exclude = {canary_cfg["vix_ticker"]}
    us_rows, kr_rows = watchlist_rows(features_dir, exclude)
    head = "<thead><tr><th>종목</th><th>추세 <small>(200일선 기준)</small></th><th>과열 여부</th><th class='num'>최근 1개월</th></tr></thead>"
    watch = "<section><h2>관심 종목 한눈에</h2>"
    for title, rows in [("미국", us_rows), ("한국", kr_rows)]:
        if rows:
            watch += f"<h3 style='font-size:15px;margin:12px 0 4px'>{title}</h3><div class='table-wrap'><table>{head}<tbody>{''.join(rows)}</tbody></table></div>"
    watch += ("<p class='note'>추세: 가격이 최근 200일 평균보다 위면 상승, 아래면 하락. "
              "과열 여부: RSI 70 이상이면 단기간에 많이 오른 상태, 30 이하면 많이 빠진 상태.</p></section>")

    # 6) 신뢰도
    trust = f"<section><h2>이 신호를 얼마나 믿어도 되나</h2>{trust_box(report_dir)}</section>"

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    page = (
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>오늘의 시장 상태</title><style>{STYLE}</style></head><body><main>"
        f"<header><h1>오늘의 시장 상태</h1><p>미국 장 마감 {as_of.date()} 기준 · 생성 {generated}</p></header>"
        f"{hero}{reasons}{history}{alloc}{watch}{trust}"
        "<p class='note'>교육·연구 목적의 자료이며 투자 권유가 아닙니다. 과거 성과가 미래 수익을 보장하지 않습니다.</p>"
        f"</main><div id='tip' role='tooltip'></div><script>{SCRIPT}</script></body></html>"
    )
    return page, as_of


def run(config: dict, tickers: list[str] | None = None) -> Path:
    """오늘의 시장 상태 리포트를 만든다. (③ indicators를 먼저 실행해야 함)

    tickers 인자는 다른 단계와 형태를 맞추기 위한 것으로, 이 단계에서는 받아 둔 모든 종목을 보여 준다.
    """
    data_cfg = config["data"]
    report_dir = PROJECT_ROOT / config["report"]["output_dir"]
    features_dir = PROJECT_ROOT / data_cfg["features_dir"]
    signals = pd.read_csv(PROJECT_ROOT / data_cfg["signals_path"], index_col="Date", parse_dates=True)

    page, as_of = build_html(config, signals, features_dir, report_dir)

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "daily").mkdir(exist_ok=True)
    today_path = report_dir / "market_today.html"
    today_path.write_text(page, encoding="utf-8")
    (report_dir / "daily" / f"market_{as_of.date()}.html").write_text(page, encoding="utf-8")

    level = int(signals.dropna(subset=["Risk_Level"])["Risk_Level"].iloc[-1])
    logger.info("오늘의 시장 상태 (%s): %s → %s", as_of.date(), risk.RISK_LABELS[level], today_path)
    return today_path
