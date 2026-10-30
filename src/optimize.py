"""⑤ 전략 개선 — 파라미터 최적화 및 Out-of-Sample 검증

설정 값 조합(그리드)을 모두 백테스트하고, '학습 기간' 성적만으로 조합을 고른 뒤
처음 보는 '검증 기간'에서도 통하는지 확인한다.

왜 기간을 나누나? (과최적화 방지)
  72가지 조합을 같은 기간에서 비교해 1등을 고르면, 그 1등은 "그 시기에 우연히 맞은 값"일 수 있다.
  시험 문제를 미리 보고 공부한 것과 같다. 고를 때 쓰지 않은 기간에서 성적이 유지돼야 진짜 개선이다.

고르는 기준
  1. 칼마 지수(CAGR ÷ |MDD|): 낙폭 대비 수익. '자산 방어'가 목적이므로 수익률만 보지 않는다.
  2. 주변 평균 칼마: 자기 자신 + 이웃 조합(주의 비중이나 VIX 기준값만 한 칸 다른 조합)의 평균.
     이웃 값에서 성적이 급락하는 '뾰족한 1등'은 우연일 가능성이 높아서, 주변까지 고르게 좋은 조합을 고른다.

계산 방식
  각 조합을 전체 기간으로 한 번 시뮬레이션한 뒤, 자산 가치 곡선을 학습/검증 구간으로 잘라 성과를 잰다.
  신호는 과거 정보만 쓰므로(하루 뒤 매매) 잘라서 재도 미래 정보가 섞이지 않는다.
"""

import itertools
import logging

import pandas as pd

from src import backtest, risk
from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

PARAM_COLS = ["방어", "주의비중", "VIX", "카나리아"]


def vix_label(threshold: float | None) -> str:
    return "없음" if threshold is None else str(threshold)


def slice_performance(values: pd.Series, orders: pd.DataFrame, start: str | None, end: str | None) -> dict:
    """자산 가치 곡선의 일부 구간 성과."""
    part = values.loc[start:end]
    n_orders = int(orders.loc[start:end].notna().any(axis=1).sum())
    return backtest.performance(part, n_orders)


def evaluate(values: pd.Series, orders: pd.DataFrame, train_end: str, test_start: str) -> dict:
    """학습·검증 구간 성과를 한 행으로."""
    train = slice_performance(values, orders, None, train_end)
    test = slice_performance(values, orders, test_start, None)
    row = {}
    for prefix, perf in [("학습", train), ("검증", test)]:
        for key in ["CAGR", "MDD", "칼마", "샤프"]:
            row[f"{prefix} {key}"] = perf[key]
    return row


def add_neighbor_score(results: pd.DataFrame, grid: dict) -> pd.DataFrame:
    """각 조합의 '주변 평균 칼마'(자신 + 주의비중·VIX가 한 칸 다른 이웃)를 계산한다."""
    caution_order = grid["caution_equity"]
    vix_order = [vix_label(v) for v in grid["vix_threshold"]]

    def neighbors_mean(row: pd.Series) -> float:
        same_group = results[(results["방어"] == row["방어"]) & (results["카나리아"] == row["카나리아"])]
        ci, vi = caution_order.index(row["주의비중"]), vix_order.index(row["VIX"])
        near = same_group[
            ((same_group["VIX"] == row["VIX"]) & same_group["주의비중"].map(lambda c: abs(caution_order.index(c) - ci) <= 1))
            | ((same_group["주의비중"] == row["주의비중"]) & same_group["VIX"].map(lambda v: abs(vix_order.index(v) - vi) <= 1))
        ]
        return near["학습 칼마"].mean()

    results = results.copy()
    results["학습 주변칼마"] = results.apply(neighbors_mean, axis=1)
    return results


def format_table(df: pd.DataFrame) -> str:
    """퍼센트·소수 컬럼을 보기 좋게."""
    shown = df.copy()
    for col in shown.columns:
        if col.endswith(("CAGR", "MDD")):
            shown[col] = shown[col].map(lambda x: f"{x:+.1%}")
        elif col.endswith(("칼마", "샤프", "주변칼마")):
            shown[col] = shown[col].map(lambda x: f"{x:.2f}")
        elif col == "주의비중":
            shown[col] = shown[col].map(lambda x: f"{x:.0%}")
    return shown.to_string(index=False)


def run(config: dict, tickers: list[str] | None = None) -> pd.DataFrame:
    """파이프라인 ⑤ 단계: 그리드 탐색 → 학습 기간으로 선택 → 검증 기간으로 확인.

    tickers 인자는 다른 단계와 형태를 맞추기 위한 것으로, 이 단계에서는 쓰지 않는다.
    """
    data_cfg, bt_cfg, uni, opt_cfg = config["data"], config["backtest"], config["universe"], config["optimize"]
    grid, train_end, test_start = opt_cfg["grid"], opt_cfg["train_end"], opt_cfg["test_start"]
    processed_dir = PROJECT_ROOT / data_cfg["processed_dir"]
    report_dir = PROJECT_ROOT / config["report"]["output_dir"]
    offensive, benchmark, rebalance = uni["offensive"], uni["benchmark"], bt_cfg["rebalance"]

    # 1) 데이터: 후보 방어 자산까지 포함한 종가 + 방식·VIX 기준별 위험 단계
    all_defensive = list(dict.fromkeys(t for group in grid["defensive"] + [uni["defensive"]] for t in group))
    closes = backtest.load_closes(list(dict.fromkeys(offensive + all_defensive + [benchmark])), processed_dir)
    canary_closes, vix_close = risk.load_canary_closes(config, processed_dir)

    current_key = (config["canary"]["method"], config["canary"]["vix_threshold"])
    level_keys = set(itertools.product(grid["canary_method"], grid["vix_threshold"])) | {current_key}
    levels = {
        (m, v): risk.compute_risk_signals(canary_closes, vix_close, v, m)["Risk_Level"] for m, v in level_keys
    }

    # 모든 방식의 신호가 준비된 날부터 시작해 같은 기간에서 비교한다
    levels_df = pd.DataFrame({k: s for k, s in levels.items()}).reindex(closes.index).dropna()
    closes = closes.loc[levels_df.index]
    logger.info(
        "기간: 학습 %s ~ %s / 검증 %s ~ %s",
        closes.index[0].date(), train_end, test_start, closes.index[-1].date(),
    )

    def run_canary(method, vix, defensive, caution) -> tuple[pd.Series, pd.DataFrame]:
        level = levels_df[(method, vix)].astype(int)
        weights = backtest.canary_weights(level, offensive, defensive, {0: 1.0, 1: caution, 2: 0.0})
        orders = backtest.to_orders(weights, rebalance)
        return backtest.simulate(closes, orders, bt_cfg).value(), orders

    # 2) 그리드 탐색
    combos = list(itertools.product(grid["defensive"], grid["caution_equity"], grid["vix_threshold"], grid["canary_method"]))
    logger.info("조합 %d개 백테스트 중...", len(combos))
    rows = []
    for defensive, caution, vix, method in combos:
        values, orders = run_canary(method, vix, defensive, caution)
        rows.append({
            "방어": "+".join(defensive), "주의비중": caution, "VIX": vix_label(vix), "카나리아": method,
            **evaluate(values, orders, train_end, test_start),
        })
    results = add_neighbor_score(pd.DataFrame(rows), grid)

    # 3) 학습 기간 성적으로만 선택 (검증 성적은 보지 않는다)
    ranked = results.sort_values("학습 주변칼마", ascending=False)
    best = ranked.iloc[0]
    train_cols = PARAM_COLS + ["학습 CAGR", "학습 MDD", "학습 칼마", "학습 주변칼마"]
    logger.info("[학습 기간] 상위 5개 조합 (주변 평균 칼마 순)\n%s", format_table(ranked[train_cols].head(5)))

    # 4) 검증: 선택한 조합 vs 방어 없음 vs 현재 설정 vs 벤치마크
    vix_by_label = {vix_label(v): v for v in grid["vix_threshold"]}
    selected = (best["방어"].split("+"), best["주의비중"], vix_by_label[best["VIX"]], best["카나리아"])
    current = (uni["defensive"], bt_cfg["equity_by_level"][1], config["canary"]["vix_threshold"], config["canary"]["method"])

    def describe(defensive, caution, vix, method) -> str:
        return f"방어={'+'.join(defensive)}, 주의={caution:.0%}, VIX={vix_label(vix)}, {method}"

    compare_rows = {}
    for name, orders in {
        f"{benchmark} 보유": backtest.to_orders(backtest.fixed_weights(closes.index, {benchmark: 1.0}), "once"),
        "방어 없음": backtest.to_orders(
            backtest.fixed_weights(closes.index, {t: 1 / len(offensive) for t in offensive}), rebalance
        ),
    }.items():
        compare_rows[name] = evaluate(backtest.simulate(closes, orders, bt_cfg).value(), orders, train_end, test_start)
    for name, (defensive, caution, vix, method) in {
        f"현재 설정 ({describe(*current)})": current,
        f"선택 조합 ({describe(*selected)})": selected,
    }.items():
        values, orders = run_canary(method, vix, defensive, caution)
        compare_rows[name] = evaluate(values, orders, train_end, test_start)
    compare = pd.DataFrame(compare_rows).T

    shown = compare[["학습 CAGR", "학습 MDD", "학습 칼마", "검증 CAGR", "검증 MDD", "검증 칼마"]]
    logger.info("[검증] 학습 기간 vs 검증 기간 성과\n%s", format_table(shown.rename_axis("전략").reset_index()))

    # 5) 결론
    sel_name, cur_name = compare.index[3], compare.index[2]
    sel_calmar = compare.loc[sel_name, "검증 칼마"]
    beats_none = sel_calmar > compare.loc["방어 없음", "검증 칼마"]
    beats_current = sel_calmar > compare.loc[cur_name, "검증 칼마"]
    if beats_none and beats_current:
        verdict = "선택 조합이 검증 기간에서도 '방어 없음'과 '현재 설정'보다 칼마가 높음 → 설정 반영을 검토할 만함"
    elif beats_current:
        verdict = "선택 조합이 현재 설정보다는 낫지만 '방어 없음'보다 칼마가 낮음 → 방어 로직의 가치가 검증되지 않음"
    else:
        verdict = "선택 조합이 검증 기간에서 현재 설정보다도 못함 → 학습 기간에만 맞은 값(과최적화), 반영하지 않음"
    logger.info("결론: %s", verdict)

    # 6) 저장
    report_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(report_dir / "optimize_results.csv", index=False)
    compare.to_csv(report_dir / "optimize_summary.csv")
    logger.info("저장: %s, %s", report_dir / "optimize_results.csv", report_dir / "optimize_summary.csv")
    return compare
