"""④ 전략 평가 — 백테스팅 및 CAGR / MDD 등 성과 지표 산출

과거 데이터로 "이 규칙대로 투자했다면 어땠을까"를 계산한다.

비교 전략 (모두 같은 기간, 같은 수수료·슬리피지)
  1. <벤치마크> 보유        : 벤치마크(SPY) 하나를 사서 계속 보유
  2. 공격자산 보유(방어 없음): 공격 자산을 같은 비중으로 보유, 리밸런싱 주기는 3번과 같음
  3. 카나리아 단검          : 위험 단계(src/risk.py)에 따라 주식 비중 조절
                              정상 → 주식 100%, 주의 → 50%, 위험 → 0% (나머지는 방어 자산)
  2와 3은 방어 로직만 다르다. 둘의 차이가 곧 '방어 로직의 효과'다.

매매 시점 (룩어헤드 편향 방지)
  t일 종가로 신호 확인 → t+1일 종가에 매매. 신호를 보고 나서 하루 뒤에 거래하므로 미래 정보를 쓰지 않는다.

성과 지표
  CAGR   : 연평균 수익률. 매년 몇 %씩 복리로 불어났는지. (최종/처음)^(1/연수) - 1
  MDD    : 최대 낙폭. 고점 대비 가장 크게 떨어졌던 비율. "최악의 순간에 얼마나 잃고 있었나"
  변동성 : 일별 수익률 표준편차 × √252. 수익률이 얼마나 출렁이는지
  샤프   : CAGR / 변동성 (무위험 수익률 0 가정). 같은 위험을 지고 얼마나 벌었는지 — 클수록 좋음
  칼마   : CAGR / |MDD|. 최악의 낙폭 대비 수익 — 클수록 좋음

한계
  - 환율은 반영하지 않는다(달러 기준 수익률).
  - 세금은 반영하지 않는다.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

from src import risk, storage
from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 데이터 준비
# ---------------------------------------------------------------------------

def load_closes(tickers: list[str], processed_dir: Path) -> pd.DataFrame:
    """여러 종목의 정제된 종가를 한 표로 합친다. 모든 종목이 거래된 날짜만 남긴다."""
    closes = {}
    for t in tickers:
        path = storage.find_path(processed_dir, t)
        if path is None:
            raise FileNotFoundError(f"[{t}] 정제 파일이 없습니다. 먼저 collect, clean 단계를 실행하세요.")
        closes[t] = storage.load_prices(path)["Close"]
    return pd.DataFrame(closes).dropna()


# ---------------------------------------------------------------------------
# 목표 비중 만들기
# ---------------------------------------------------------------------------

def canary_weights(risk_level: pd.Series, offensive: list[str], defensive: list[str], equity_by_level: dict) -> pd.DataFrame:
    """위험 단계별 목표 비중. 주식 몫은 공격 자산끼리, 나머지는 방어 자산끼리 똑같이 나눈다."""
    equity = risk_level.map(equity_by_level)
    weights = {t: equity / len(offensive) for t in offensive}
    weights.update({t: (1 - equity) / len(defensive) for t in defensive})
    return pd.DataFrame(weights)


def fixed_weights(index: pd.DatetimeIndex, allocation: dict[str, float]) -> pd.DataFrame:
    """항상 같은 비중."""
    return pd.DataFrame({t: w for t, w in allocation.items()}, index=index)


def to_orders(target: pd.DataFrame, rebalance: str) -> pd.DataFrame:
    """목표 비중을 '언제 주문을 낼지' 표로 바꾼다. 주문이 없는 날은 NaN.

    rebalance:
      'monthly' : 매월 마지막 거래일 신호로 비중을 정한다 (그사이 신호 변화는 무시)
      'daily'   : 목표 비중이 바뀐 날에만 주문한다
      'once'    : 첫날 한 번만 사고 계속 보유
    공통: 신호 확인 다음 날 매매하도록 하루 뒤로 민다(shift(1)).
    """
    if rebalance == "once":
        orders = target.iloc[[0]].reindex(target.index)
        return orders

    if rebalance == "monthly":
        month_end = target.index.to_series().groupby(target.index.to_period("M")).transform("max") == target.index
        decided = target.where(month_end, np.nan)
    elif rebalance == "daily":
        changed = target.ne(target.shift()).any(axis=1)
        decided = target.where(changed, np.nan)
    else:
        raise ValueError(f"알 수 없는 리밸런싱 방식: {rebalance}")

    orders = decided.shift(1)
    orders.iloc[0] = target.iloc[0]  # 시작일에는 첫 목표 비중으로 진입
    return orders


# ---------------------------------------------------------------------------
# 시뮬레이션과 성과 지표
# ---------------------------------------------------------------------------

def simulate(closes: pd.DataFrame, orders: pd.DataFrame, cfg: dict) -> vbt.Portfolio:
    """vectorbt로 포트폴리오를 시뮬레이션한다.

    주문은 해당 날짜 종가에 '목표 비중'만큼 맞추도록 사고판다. 수수료·슬리피지는 거래 금액에 비례해 뺀다.
    """
    orders = orders.reindex(columns=closes.columns)
    # 주문하는 날에는 이 전략에 없는 종목의 목표 비중을 0으로 둔다 (주문 없는 날은 그대로 NaN)
    order_days = orders.notna().any(axis=1)
    orders.loc[order_days] = orders.loc[order_days].fillna(0)
    return vbt.Portfolio.from_orders(
        close=closes,
        size=orders,
        size_type="targetpercent",
        group_by=True,          # 여러 종목을 하나의 포트폴리오로 묶기
        cash_sharing=True,      # 현금을 공유
        call_seq="auto",        # 파는 주문을 먼저 처리해 살 돈 마련
        fees=cfg["fee_rate"],
        slippage=cfg["slippage"],
        init_cash=cfg["initial_cash"],
        freq="1D",
    )


def max_drawdown(value: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """최대 낙폭과 그때의 고점·저점 날짜."""
    drawdown = value / value.cummax() - 1
    trough = drawdown.idxmin()
    peak = value.loc[:trough].idxmax()
    return float(drawdown.min()), peak, trough


def performance(value: pd.Series, n_orders: int) -> dict:
    """자산 가치 곡선으로 성과 지표를 계산한다."""
    years = (value.index[-1] - value.index[0]).days / 365.25
    cagr = (value.iloc[-1] / value.iloc[0]) ** (1 / years) - 1
    daily = value.pct_change().dropna()
    vol = daily.std() * np.sqrt(TRADING_DAYS)
    mdd, peak, trough = max_drawdown(value)
    return {
        "CAGR": cagr,
        "MDD": mdd,
        "MDD 기간": f"{peak.date()} ~ {trough.date()}",
        "변동성": vol,
        "샤프": cagr / vol if vol else np.nan,
        "칼마": cagr / abs(mdd) if mdd else np.nan,
        "최종 배수": value.iloc[-1] / value.iloc[0],
        "매매 횟수": n_orders,
    }


def yearly_returns(values: pd.DataFrame) -> pd.DataFrame:
    """연도별 수익률 표."""
    year_end = values.groupby(values.index.year).last()
    prev = year_end.shift(1)
    prev.iloc[0] = values.iloc[0]
    return year_end / prev - 1


# ---------------------------------------------------------------------------
# 출력 도우미
# ---------------------------------------------------------------------------

def format_summary(summary: pd.DataFrame) -> str:
    """성과 표를 보기 좋은 문자열로."""
    shown = summary.copy()
    for col in ["CAGR", "MDD", "변동성"]:
        shown[col] = shown[col].map(lambda x: f"{x:+.1%}" if col != "변동성" else f"{x:.1%}")
    for col in ["샤프", "칼마"]:
        shown[col] = shown[col].map(lambda x: f"{x:.2f}")
    shown["최종 배수"] = shown["최종 배수"].map(lambda x: f"{x:.2f}배")
    return shown.to_string()


def describe_defense(risk_level: pd.Series) -> str:
    """방어 로직이 얼마나 자주, 언제 발동했는지 요약."""
    share = risk_level.value_counts(normalize=True).sort_index()
    share_text = ", ".join(f"{risk.RISK_LABELS[int(k)]} {v:.0%}" for k, v in share.items())

    # 위험(2) 구간 중 길었던 것 (월말 기준이 아닌 일별 신호 기준)
    is_danger = risk_level == 2
    block_id = (is_danger != is_danger.shift()).cumsum()
    blocks = risk_level[is_danger].groupby(block_id[is_danger]).agg(lambda s: (s.index[0], s.index[-1], len(s)))
    longest = sorted(blocks, key=lambda b: -b[2])[:5]
    longest_text = "\n".join(f"    {s.date()} ~ {e.date()} ({n}거래일)" for s, e, n in sorted(longest))
    return f"  기간 비율: {share_text}\n  가장 길었던 '위험' 구간:\n{longest_text}"


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------

def backtest_market(market: str, uni: dict, signals: pd.DataFrame, bt_cfg: dict, processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """한 시장(미국 또는 한국)에서 세 전략을 백테스트하고 결과를 로그로 출력한다.

    Args:
        market: 'US' 또는 'KR'. 한국이면 전날 미국 위험 신호를 쓴다(src/risk.py align_to_ticker).
        uni: {'offensive': [...], 'defensive': [...], 'benchmark': ...}

    Returns:
        (성과 요약표, 날짜별 자산 가치)
    """
    offensive, defensive, benchmark = uni["offensive"], uni["defensive"], uni["benchmark"]

    # 1) 데이터: 종가 + 위험 신호를 같은 날짜로 맞추고, 위험 신호가 나오는 날부터 시작
    closes = load_closes(list(dict.fromkeys(offensive + defensive + [benchmark])), processed_dir)
    aligned = risk.align_to_ticker(signals, closes.index, korean=(market == "KR"))
    data = closes.join(aligned["Risk_Level"]).dropna()
    closes, risk_level = data[closes.columns], data["Risk_Level"].astype(int)
    logger.info("[%s] 백테스트 기간: %s ~ %s (%d거래일)", market, closes.index[0].date(), closes.index[-1].date(), len(closes))

    # 2) 전략별 목표 비중 → 주문표
    rebalance = bt_cfg["rebalance"]
    equity_by_level = {int(k): v for k, v in bt_cfg["equity_by_level"].items()}
    strategies = {
        f"{benchmark} 보유": to_orders(fixed_weights(closes.index, {benchmark: 1.0}), "once"),
        "공격자산 보유(방어 없음)": to_orders(
            fixed_weights(closes.index, {t: 1 / len(offensive) for t in offensive}), rebalance
        ),
        "카나리아 단검": to_orders(canary_weights(risk_level, offensive, defensive, equity_by_level), rebalance),
    }

    # 3) 시뮬레이션 및 성과 계산
    values, rows = {}, {}
    for name, orders in strategies.items():
        pf = simulate(closes, orders, bt_cfg)
        values[name] = pf.value()
        rows[name] = performance(values[name], int(orders.notna().any(axis=1).sum()))
    values = pd.DataFrame(values)
    summary = pd.DataFrame(rows).T

    # 4) 결과 출력
    logger.info("[%s] 성과 요약\n%s", market, format_summary(summary))
    logger.info("[%s] 방어 로직 발동 현황\n%s", market, describe_defense(risk_level))
    logger.info("[%s] 연도별 수익률\n%s", market, yearly_returns(values).map(lambda x: f"{x:+.1%}").to_string())
    return summary, values


def run(config: dict, tickers: list[str] | None = None) -> pd.DataFrame:
    """파이프라인 ④ 단계: 시장별로 세 전략을 백테스트하고 결과를 reports/에 저장한다.

    - 미국: 설정의 universe
    - 한국: 설정의 universe_kr (없으면 건너뜀)
    tickers 인자는 다른 단계와 형태를 맞추기 위한 것으로, 이 단계에서는 설정 파일의 자산 구성을 쓴다.
    """
    data_cfg, bt_cfg = config["data"], config["backtest"]
    processed_dir = PROJECT_ROOT / data_cfg["processed_dir"]
    report_dir = PROJECT_ROOT / config["report"]["output_dir"]
    signals = pd.read_csv(PROJECT_ROOT / data_cfg["signals_path"], index_col="Date", parse_dates=True)

    logger.info("리밸런싱: %s, 수수료 %.3f%%, 슬리피지 %.2f%%", bt_cfg["rebalance"], bt_cfg["fee_rate"] * 100, bt_cfg["slippage"] * 100)
    markets = {"US": config["universe"]}
    if "universe_kr" in config:
        markets["KR"] = config["universe_kr"]

    report_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for market, uni in markets.items():
        summary, values = backtest_market(market, uni, signals, bt_cfg, processed_dir)
        summaries.append(summary.rename_axis("전략").reset_index().assign(시장=market))
        values.to_csv(report_dir / f"backtest_equity_{market.lower()}.csv")

    summary = pd.concat(summaries).set_index(["시장", "전략"])
    summary.to_csv(report_dir / "backtest_summary.csv")
    logger.info("저장: %s, backtest_equity_<시장>.csv", report_dir / "backtest_summary.csv")
    return summary
