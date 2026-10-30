"""결과 그래프 — ④ 백테스트와 ⑤ 전략 개선 결과를 그림으로 저장한다 (reports/*.png)

1. chart_backtest_<시장>.png
     위: 1원이 몇 원이 됐는지 (자산 곡선, 로그 눈금) + '위험' 신호 구간 음영
     아래: 고점 대비 얼마나 떨어져 있었는지 (낙폭)
2. chart_defensive.png
     방어 자산별 칼마(낙폭 대비 수익)가 학습 기간 → 검증 기간에 어떻게 바뀌었는지 (기울기 차트)
     선이 서로 엇갈릴수록 '과거 순위가 미래에 통하지 않았다'는 뜻이다.

로그 눈금: 세로축 한 칸이 '2배'처럼 같은 비율을 뜻한다. 오래 복리로 불어난 곡선을 공정하게 비교할 수 있다.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from src import risk
from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# 색상 (색각 이상 검사를 통과한 순서. 전략마다 색을 고정한다)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
DANGER_BAND = "#ecebe7"     # '위험' 신호 구간 음영 (중립 회색)
MUTED_LINE = "#b9b7b0"      # 강조하지 않는 선
COLOR_CANARY = "#2a78d6"    # 카나리아 단검 (파랑)
COLOR_NO_DEFENSE = "#eb6834"  # 방어 없음 (주황)
COLOR_BUY_HOLD = "#1baf7a"  # 단순 보유 (청록)

# 한글 폰트 후보 (WSL은 Windows 폰트를 쓸 수 있다)
KOREAN_FONTS = [
    "/mnt/c/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]


def setup_style() -> None:
    """한글 폰트와 기본 스타일(얇은 선, 흐린 격자)을 설정한다."""
    for path in KOREAN_FONTS:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            bold = Path(path.replace("malgun.ttf", "malgunbd.ttf"))  # 굵은 글씨용 (있으면)
            if bold.exists() and bold != Path(path):
                font_manager.fontManager.addfont(str(bold))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    else:
        logger.warning("한글 폰트를 찾지 못했습니다. 그래프의 한글이 깨질 수 있습니다.")

    plt.rcParams.update({
        "axes.unicode_minus": False,  # 한글 폰트에서 마이너스 기호 깨짐 방지
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })


def strategy_color(name: str) -> str:
    if "카나리아" in name:
        return COLOR_CANARY
    if "방어 없음" in name:
        return COLOR_NO_DEFENSE
    return COLOR_BUY_HOLD


def drop_duplicate_columns(values: pd.DataFrame) -> pd.DataFrame:
    """완전히 같은 곡선은 하나만 남긴다. (예: 한국은 공격 자산이 1개라 '보유'와 '방어 없음'이 같다)"""
    keep = []
    for col in values.columns:
        if not any((values[col] - values[k]).abs().max() < 1e-6 * values[k].abs().max() for k in keep):
            keep.append(col)
    return values[keep]


def danger_spans(level: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """위험 단계(2)가 이어진 구간의 (시작, 끝) 목록."""
    is_danger = level == 2
    block = (is_danger != is_danger.shift()).cumsum()
    return [(g.index[0], g.index[-1]) for _, g in level[is_danger].groupby(block[is_danger])]


def plot_backtest(values: pd.DataFrame, level: pd.Series, summary: pd.DataFrame, title: str, path: Path) -> None:
    """자산 곡선(위)과 낙폭(아래)."""
    values = drop_duplicate_columns(values)
    growth = values / values.iloc[0]
    drawdown = values / values.cummax() - 1

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08}
    )

    # 위험 신호 구간 음영 (두 그래프 모두)
    for start, end in danger_spans(level.reindex(values.index)):
        for ax in (ax_top, ax_bottom):
            ax.axvspan(start, end, color=DANGER_BAND, linewidth=0, zorder=0)

    for col in growth.columns:
        color = strategy_color(col)
        ax_top.plot(growth.index, growth[col], color=color, linewidth=1.6, zorder=2)
        ax_bottom.plot(drawdown.index, drawdown[col], color=color, linewidth=1.2, zorder=2)

    # 선 끝에 직접 이름 표시 (범례 대신 바로 읽히도록)
    ends = growth.iloc[-1].sort_values()
    last_y = None
    for col, y in ends.items():
        label_y = y if last_y is None else max(y, last_y * 1.12)  # 이름이 겹치지 않게 살짝 띄움
        perf = summary.loc[col]
        ax_top.annotate(
            f"{col}\n{y:.1f}배 · CAGR {perf['CAGR']:+.1%} · MDD {perf['MDD']:.1%}",
            xy=(growth.index[-1], label_y), xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=9, color=TEXT,
        )
        ax_top.plot([growth.index[-1]], [y], marker="o", markersize=5, color=strategy_color(col),
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        last_y = label_y

    ax_top.set_yscale("log")
    ax_top.yaxis.set_major_locator(LogLocator(base=2))
    ax_top.yaxis.set_minor_formatter(NullFormatter())
    ax_top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}배"))
    ax_top.set_ylabel("1원이 몇 원이 됐나 (로그 눈금)")
    ax_top.set_title(title, loc="left", fontsize=14, color=TEXT, pad=26)
    ax_top.text(0, 1.02, "회색 음영 = 카나리아 '위험' 신호 구간 (일별 신호, 매매는 월말 기준)",
                transform=ax_top.transAxes, fontsize=9, color=TEXT_SECONDARY)

    ax_bottom.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_bottom.set_ylabel("고점 대비 낙폭")

    fig.subplots_adjust(left=0.08, right=0.74, top=0.9, bottom=0.06)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def spread_labels(ys: dict[str, float], min_gap: float) -> dict[str, float]:
    """이름표가 겹치지 않도록 세로 위치를 min_gap 이상 벌린다. (위치 순서는 유지)"""
    items = sorted(ys.items(), key=lambda kv: kv[1])
    placed: dict[str, float] = {}
    prev = None
    for name, y in items:
        y = y if prev is None else max(y, prev + min_gap)
        placed[name] = prev = y
    # 전체가 위로 밀린 만큼 가운데로 되돌린다
    shift = (sum(placed.values()) - sum(ys.values())) / len(ys)
    return {name: y - shift for name, y in placed.items()}


def plot_defensive(results: pd.DataFrame, no_defense: pd.Series, path: Path, highlight: dict[str, str]) -> None:
    """방어 자산별 칼마의 학습 → 검증 변화 (기울기 차트).

    각 방어 자산마다 나머지 설정(주의 비중·VIX·판단 방식)을 바꾼 24개 조합의 칼마 중앙값을 쓴다.
    """
    median = results.groupby("방어")[["학습 칼마", "검증 칼마"]].median()
    xs = [0, 1]

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # 기준선: 방어 없음
    baseline = "방어 없음 (기준)"
    ax.plot(xs, [no_defense["학습 칼마"], no_defense["검증 칼마"]], color=TEXT_SECONDARY,
            linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)

    # 강조하지 않는 자산을 먼저, 강조 자산을 위에 그린다
    order = [n for n in median.index if n not in highlight] + [n for n in median.index if n in highlight]
    for name in order:
        y0, y1 = median.loc[name, "학습 칼마"], median.loc[name, "검증 칼마"]
        ax.plot(xs, [y0, y1], color=highlight.get(name, MUTED_LINE), linewidth=2.2 if name in highlight else 1.2,
                zorder=3 if name in highlight else 2, marker="o", markersize=7 if name in highlight else 5,
                markeredgecolor=SURFACE, markeredgewidth=1.5)

    # 양쪽 이름표: 겹치지 않게 벌려서 배치
    all_values = pd.concat([median["학습 칼마"], median["검증 칼마"]])
    min_gap = (all_values.max() - all_values.min()) * 0.045
    for x, col, side in [(0, "학습 칼마", "left"), (1, "검증 칼마", "right")]:
        ys = {**median[col].to_dict(), baseline: no_defense[col]}
        for name, label_y in spread_labels(ys, min_gap).items():
            value = ys[name]
            if name == baseline:
                text, color, weight = baseline, TEXT_SECONDARY, "normal"
            else:
                text = f"{name} {value:.2f}" if side == "left" else f"{value:.2f} {name}"
                color, weight = (TEXT, "bold") if name in highlight else (TEXT_SECONDARY, "normal")
            ax.annotate(text, xy=(x, value), xytext=(x + (-0.04 if side == "left" else 0.04), label_y),
                        textcoords="data", ha="right" if side == "left" else "left", va="center",
                        fontsize=9, color=color, fontweight=weight)

    ax.set_xticks(xs, ["학습 기간\n2011–2018", "검증 기간\n2019–현재"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("칼마 (CAGR ÷ 최대 낙폭, 높을수록 좋음)")
    ax.grid(axis="x", visible=False)
    ax.set_title("방어 자산 순위가 뒤집혔다", loc="left", fontsize=14, color=TEXT, pad=26)
    ax.text(0, 1.02, "방어 자산별 칼마 중앙값 (나머지 설정 24개 조합 기준)",
            transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY)

    fig.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.1)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(config: dict, tickers: list[str] | None = None) -> list[Path]:
    """④·⑤ 결과 파일을 읽어 그래프를 reports/에 저장한다. (④ backtest, ⑤ optimize를 먼저 실행해야 함)"""
    setup_style()
    data_cfg = config["data"]
    report_dir = PROJECT_ROOT / config["report"]["output_dir"]
    signals = pd.read_csv(PROJECT_ROOT / data_cfg["signals_path"], index_col="Date", parse_dates=True)
    summary = pd.read_csv(report_dir / "backtest_summary.csv", index_col=["시장", "전략"])
    titles = {"US": "미국: 카나리아 단검 vs 단순 보유", "KR": "한국: 카나리아 단검 vs KODEX 200 보유"}

    saved = []
    for market in summary.index.get_level_values("시장").unique():
        values = pd.read_csv(report_dir / f"backtest_equity_{market.lower()}.csv", index_col=0, parse_dates=True)
        level = risk.align_to_ticker(signals, values.index, korean=(market == "KR"))["Risk_Level"]
        path = report_dir / f"chart_backtest_{market.lower()}.png"
        plot_backtest(values, level, summary.loc[market], titles.get(market, market), path)
        saved.append(path)

    optimize_results = report_dir / "optimize_results.csv"
    if optimize_results.exists():
        results = pd.read_csv(optimize_results)
        no_defense = pd.read_csv(report_dir / "optimize_summary.csv", index_col=0).loc["방어 없음"]
        path = report_dir / "chart_defensive.png"
        plot_defensive(results, no_defense, path, highlight={"IEF": COLOR_NO_DEFENSE, "GLD": COLOR_CANARY})
        saved.append(path)

    for p in saved:
        logger.info("저장: %s", p)
    return saved
