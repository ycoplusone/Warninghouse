"""③ 데이터 분석 — SMA, MACD, RSI, 볼린저 밴드 등 지표 계산

data/processed/의 정제 데이터에 지표와 위험 신호(src/risk.py)를 붙여 data/features/에 저장한다.
파일 이름 규칙은 같다. (예: data/features/005930_삼성전자.csv)

지표 설명 (모두 종가 기준)
  SMA (단순 이동평균선)
      최근 N일 종가의 평균. 가격이 SMA 위에 있으면 상승 추세, 아래면 하락 추세로 본다.
      짧은 SMA(50일)가 긴 SMA(200일)를 위로 뚫으면 '골든크로스'(상승 신호), 아래로 뚫으면 '데드크로스'.
  MACD (이동평균 수렴·확산)
      12일 지수이동평균 - 26일 지수이동평균. 이 값의 9일 평균을 '시그널'이라 한다.
      MACD가 시그널 위면 상승 힘이 강해지는 중, 아래면 약해지는 중.
  RSI (상대강도지수, 0~100)
      최근 14일 동안 오른 폭과 내린 폭의 비율. 70 이상은 '과매수'(많이 올라 쉬어갈 수 있음),
      30 이하는 '과매도'(많이 빠져 반등할 수 있음)로 본다.
  볼린저 밴드
      20일 SMA(중심선) ± 2 × 20일 표준편차. 가격의 95% 정도가 밴드 안에서 움직인다.
      %B = 밴드 안에서의 위치 (0 = 하단, 0.5 = 중심, 1 = 상단). 1보다 크면 상단 돌파.

주의: 여기서 만든 지표·신호는 '그날 종가까지의 정보'로 계산한 값이다.
      백테스트에서 매매에 쓸 때는 반드시 하루 뒤로 밀어서(shift(1)) 다음 날 거래에 적용한다.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import risk, storage
from src.collect import is_korean_ticker
from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 지표 계산 (입력: 종가 Series → 출력: Series/DataFrame)
# ---------------------------------------------------------------------------

def sma(close: pd.Series, window: int) -> pd.Series:
    """단순 이동평균. 처음 window-1일은 데이터가 모자라 NaN이다."""
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    """지수 이동평균. 최근 가격에 더 큰 가중치를 준다."""
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """MACD, 시그널, 히스토그램(MACD - 시그널)."""
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"MACD": line, "MACD_Signal": signal_line, "MACD_Hist": line - signal_line})


def rsi(close: pd.Series, period: int) -> pd.Series:
    """RSI (Wilder 방식). 평균 상승폭 / 평균 하락폭으로 계산한다."""
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    result = 100 - 100 / (1 + rs)
    return result.where(avg_loss != 0, 100.0).where(avg_gain.notna())  # 하락이 전혀 없으면 100


def bollinger(close: pd.Series, window: int, num_std: float) -> pd.DataFrame:
    """볼린저 밴드 중심선·상단·하단과 %B."""
    mid = sma(close, window)
    std = close.rolling(window).std(ddof=0)
    upper, lower = mid + num_std * std, mid - num_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"BB_Mid": mid, "BB_Upper": upper, "BB_Lower": lower, "BB_PctB": pct_b})


def add_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """정제된 시세에 모든 지표 컬럼을 붙인다.

    추가 컬럼:
      SMA_<짧은>, SMA_<긴>, MACD, MACD_Signal, MACD_Hist, RSI, BB_Mid, BB_Upper, BB_Lower, BB_PctB
      Trend_Up : 종가 > 긴 SMA 이면 1 (상승 추세)
      Golden   : 짧은 SMA > 긴 SMA 이면 1 (골든크로스 상태)
      MACD_Up  : MACD > 시그널 이면 1
    """
    close = df["Close"]
    short, long = cfg["sma_short"], cfg["sma_long"]

    out = df.copy()
    out[f"SMA_{short}"] = sma(close, short)
    out[f"SMA_{long}"] = sma(close, long)
    out = out.join(macd(close, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"]))
    out["RSI"] = rsi(close, cfg["rsi_period"])
    out = out.join(bollinger(close, cfg["bollinger_period"], cfg["bollinger_std"]))

    # 0/1 신호. 비교할 지표가 아직 없으면(NaN) 신호도 NaN으로 둔다.
    def flag(cond: pd.Series, *needed: str) -> pd.Series:
        return cond.astype(float).where(out[list(needed)].notna().all(axis=1))

    out["Trend_Up"] = flag(close > out[f"SMA_{long}"], f"SMA_{long}")
    out["Golden"] = flag(out[f"SMA_{short}"] > out[f"SMA_{long}"], f"SMA_{short}", f"SMA_{long}")
    out["MACD_Up"] = flag(out["MACD"] > out["MACD_Signal"], "MACD", "MACD_Signal")
    return out


# ---------------------------------------------------------------------------
# 요약 (가장 최근 날짜의 상태를 사람이 읽기 쉬운 문장으로)
# ---------------------------------------------------------------------------

def describe_latest(df: pd.DataFrame, cfg: dict) -> str:
    """마지막 날의 지표를 해석한 한 줄 요약."""
    last = df.iloc[-1]
    long = cfg["sma_long"]
    parts = [f"{df.index[-1].date()} 종가 {last['Close']:,.2f}"]

    if pd.notna(last["Trend_Up"]):
        parts.append(f"{long}일선 {'위(상승 추세)' if last['Trend_Up'] else '아래(하락 추세)'}")
    if pd.notna(last["Golden"]):
        parts.append("골든크로스 상태" if last["Golden"] else "데드크로스 상태")
    if pd.notna(last["MACD_Up"]):
        parts.append(f"MACD {'상승' if last['MACD_Up'] else '하락'} 흐름")
    if pd.notna(last["RSI"]):
        state = "과매수" if last["RSI"] >= 70 else "과매도" if last["RSI"] <= 30 else "중립"
        parts.append(f"RSI {last['RSI']:.0f}({state})")
    if pd.notna(last["BB_PctB"]):
        parts.append(f"볼린저 %B {last['BB_PctB']:.2f}")
    if pd.notna(last.get("Risk_Level")):
        parts.append(f"시장 위험 {risk.RISK_LABELS[int(last['Risk_Level'])]}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------

def analyze_ticker(ticker: str, processed_dir: Path, features_dir: Path, cfg: dict, signals: pd.DataFrame) -> Path:
    """한 종목에 지표와 위험 신호를 붙여 저장한다."""
    path = storage.find_path(processed_dir, ticker)
    if path is None:
        raise FileNotFoundError(f"정제 파일이 없습니다. 먼저 실행하세요: python main.py --step clean --ticker {ticker}")

    df = add_indicators(storage.load_prices(path), cfg)
    df = df.join(risk.align_to_ticker(signals, df.index, korean=is_korean_ticker(ticker)))

    out_path = storage.save_prices(df, features_dir, ticker, storage.name_from_path(path, ticker))
    logger.info("[%s] %s", ticker, describe_latest(df, cfg))
    return out_path


def run(config: dict, tickers: list[str] | None = None) -> dict[str, Path]:
    """파이프라인 ③ 단계: 지표와 위험 신호를 계산해 data/features/에 저장한다.

    Args:
        config: 설정 딕셔너리
        tickers: 분석할 티커 목록. None이면 data/processed/에 있는 모든 종목.
    """
    data_cfg = config["data"]
    processed_dir = PROJECT_ROOT / data_cfg["processed_dir"]
    features_dir = PROJECT_ROOT / data_cfg["features_dir"]
    signals_path = PROJECT_ROOT / data_cfg["signals_path"]
    tickers = tickers or storage.list_tickers(processed_dir)

    # 시장 전체 위험 신호는 한 번만 계산해 모든 종목이 함께 쓴다
    signals = risk.load_risk_signals(config, processed_dir)
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(signals_path)

    last = signals.dropna(subset=["Risk_Level"]).iloc[-1]
    scores = ", ".join(f"{c[6:]} {last[c]:+.3f}" for c in signals.columns if c.startswith("Score_"))
    logger.info(
        "시장 위험 신호 (%s): %s — 카나리아 점수 [%s], VIX %.1f",
        last.name.date(), risk.RISK_LABELS[int(last["Risk_Level"])], scores, last["VIX"],
    )

    logger.info("분석 시작: %d개 종목", len(tickers))
    saved: dict[str, Path] = {}
    for ticker in tickers:
        try:
            saved[ticker] = analyze_ticker(ticker, processed_dir, features_dir, config["indicators"], signals)
        except Exception as e:
            logger.error("[%s] 분석 실패: %s", ticker, e)
    return saved
