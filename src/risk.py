"""카나리아 단검 — VIX 급등·카나리아 자산 하락 감지 리스크 관리 로직

'탄광 속 카나리아'처럼, 시장 전체가 무너지기 전에 먼저 약해지는 자산을 관찰해 위험을 미리 감지한다.
  - EEM(신흥국 주식) : 위험자산 선호가 줄면 가장 먼저 빠지는 경향
  - HYG(하이일드 채권): 기업 부도 위험이 커지면 먼저 빠지는 경향
  - AGG(미국 종합채권): 금리가 빠르게 오르는 등 채권시장 전체가 흔들리면 빠지는 경향
  - VIX(변동성 지수) : 시장 공포가 커지면 급등
  - VIX 기간구조(VIX ÷ VIX3M): '오늘의 공포'(30일 예상 변동성)를 '석 달 뒤까지의 공포'(VIX3M)로 나눈 값
      평소에는 1보다 작다(지금보다 먼 미래가 더 불확실하다고 보는 게 정상).
      1을 넘으면(이른바 '역전') 시장이 지금 당장 공포에 빠졌다는 뜻이다.
      VIX가 원래 낮던 시기에도 갑자기 튀면 잡아내므로, '30 초과' 같은 고정 기준을 보완한다.

카나리아 점수 (설정 canary.method로 선택, 둘 다 점수가 0보다 작으면 '하락 추세')
  momentum : 13612W 모멘텀 ('DAA' 전략에서 쓰는 방식)
             점수 = (12 × 1개월 수익률 + 4 × 3개월 수익률 + 2 × 6개월 수익률 + 1 × 12개월 수익률) / 4
             최근 수익률에 큰 가중치를 줘서 추세 변화를 빨리 알아챈다. (1개월 = 21거래일)
  sma200   : 200일 이동평균선과의 거리
             점수 = 종가 / 200일선 - 1. 가격이 200일선 아래면 음수. 모멘텀보다 느리고 덜 예민하다.

위험 단계 (Risk_Level)
    0 = 정상 : 카나리아 모두 상승 추세, VIX 안정
    1 = 주의 : 카나리아 일부 하락 추세
    2 = 위험 : 카나리아 모두 하락 추세, 또는 VIX가 기준값 초과
    VIX 기간구조 경보(Term_Alert)가 켜지면 단계를 최소 canary.vix_term_level(기본 1=주의)로 올린다.
"""

import logging
from pathlib import Path

import pandas as pd

from src import storage

logger = logging.getLogger(__name__)

# (기간 거래일 수, 가중치) — 1·3·6·12개월
MOMENTUM_WEIGHTS = [(21, 12), (63, 4), (126, 2), (252, 1)]

RISK_LABELS = {0: "정상", 1: "주의", 2: "위험"}


def momentum_13612w(close: pd.Series) -> pd.Series:
    """13612W 모멘텀 점수. 12개월치 데이터가 쌓이기 전까지는 NaN이다."""
    score = sum(weight * (close / close.shift(days) - 1) for days, weight in MOMENTUM_WEIGHTS)
    return score / 4


def sma200_distance(close: pd.Series) -> pd.Series:
    """종가가 200일 이동평균선보다 몇 % 위(+)/아래(-)에 있는지."""
    return close / close.rolling(200).mean() - 1


CANARY_SCORES = {"momentum": momentum_13612w, "sma200": sma200_distance}


def vix_term_ratio(vix_close: pd.Series, vix3m_close: pd.Series) -> pd.Series:
    """VIX 기간구조 비율 = VIX ÷ VIX3M. 1보다 크면 '단기 공포가 장기보다 큰' 역전 상태."""
    return vix_close / vix3m_close


def compute_risk_signals(
    canary_closes: dict[str, pd.Series],
    vix_close: pd.Series,
    vix_threshold: float | None,
    method: str = "momentum",
    vix3m_close: pd.Series | None = None,
    term_threshold: float | None = None,
    term_level: int = 1,
) -> pd.DataFrame:
    """카나리아 자산과 VIX로 날짜별 위험 신호를 계산한다.

    Args:
        canary_closes: {티커: 종가 Series}
        vix_close: VIX 종가 Series
        vix_threshold: 이 값을 넘으면 위험으로 본다. None이면 VIX 경보를 쓰지 않는다.
        method: 카나리아 점수 방식 ('momentum' 또는 'sma200')
        vix3m_close: VIX3M 종가 Series. None이면 기간구조 경보를 쓰지 않는다.
        term_threshold: VIX÷VIX3M이 이 값을 넘으면 기간구조 경보. None이면 쓰지 않는다.
        term_level: 기간구조 경보가 켜졌을 때 최소 위험 단계 (1=주의, 2=위험)

    Returns:
        날짜 인덱스, 컬럼:
          Score_<티커> : 카나리아별 점수 (음수 = 하락 추세)
          Canary_Bad   : 점수가 음수인 카나리아 개수
          VIX          : VIX 종가
          VIX_Alert    : VIX가 기준값 초과면 1
          VIX_Term     : VIX ÷ VIX3M (vix3m_close가 있을 때만)
          Term_Alert   : VIX_Term이 기준값 초과면 1 (vix3m_close가 있을 때만)
          Risk_Level   : 0(정상) / 1(주의) / 2(위험). 점수 계산 전(처음 약 1년)은 NaN
    """
    score_fn = CANARY_SCORES[method]

    # 카나리아·VIX를 같은 날짜 기준으로 맞춘다 (미국 자산끼리라 거래일이 거의 같다)
    scores = pd.DataFrame({f"Score_{t}": score_fn(c) for t, c in canary_closes.items()})
    signals = scores.join(vix_close.rename("VIX"), how="inner").dropna(subset=["VIX"])

    score_cols = list(scores.columns)
    ready = signals[score_cols].notna().all(axis=1)  # 모든 카나리아의 점수가 계산된 날

    signals["Canary_Bad"] = (signals[score_cols] < 0).sum(axis=1)
    alert = signals["VIX"] > vix_threshold if vix_threshold is not None else False
    signals["VIX_Alert"] = pd.Series(alert, index=signals.index).astype(int)

    n = len(score_cols)
    level = signals["Canary_Bad"].map(lambda bad: 0 if bad == 0 else (2 if bad == n else 1))
    level[signals["VIX_Alert"] == 1] = 2

    # VIX 기간구조: 역전(비율 > 기준)이면 단계를 최소 term_level로 올린다 (이미 더 높으면 그대로)
    if vix3m_close is not None:
        signals["VIX_Term"] = vix_term_ratio(signals["VIX"], vix3m_close.reindex(signals.index))
        term_alert = signals["VIX_Term"] > term_threshold if term_threshold is not None else False
        signals["Term_Alert"] = pd.Series(term_alert, index=signals.index).astype(int)
        level[signals["Term_Alert"] == 1] = level[signals["Term_Alert"] == 1].clip(lower=term_level)

    signals["Risk_Level"] = level.where(ready)
    return signals


def align_to_ticker(signals: pd.DataFrame, dates: pd.DatetimeIndex, korean: bool) -> pd.DataFrame:
    """위험 신호를 종목의 거래일에 맞춘다.

    - 미국 종목: 같은 날짜의 신호를 쓴다. (그날 장 마감 후 함께 확정됨)
    - 한국 종목: '전날 이전' 미국 신호를 쓴다.
        한국 날짜 t의 장이 끝나는 시점(15:30 KST)에는 미국 t-1일 종가까지만 알 수 있기 때문이다.
        같은 날짜 미국 신호를 쓰면 아직 오지 않은 미래 정보를 쓰는 셈(룩어헤드 편향)이 된다.
    - 휴장일이 달라 날짜가 비면 가장 최근 신호를 그대로 쓴다.
    """
    left = pd.DataFrame(index=dates).rename_axis("Date").reset_index()
    right = signals.rename_axis("Date").reset_index()
    merged = pd.merge_asof(left, right, on="Date", allow_exact_matches=not korean)
    return merged.set_index("Date")


def load_canary_closes(config: dict, processed_dir: Path) -> tuple[dict[str, pd.Series], pd.Series]:
    """정제된 카나리아·VIX 종가를 읽는다. Returns: ({카나리아 티커: 종가}, VIX 종가)"""
    canary_cfg = config["canary"]

    def close_of(ticker: str) -> pd.Series:
        path = storage.find_path(processed_dir, ticker)
        if path is None:
            raise FileNotFoundError(f"[{ticker}] 정제 파일이 없습니다. 먼저 collect, clean 단계를 실행하세요.")
        return storage.load_prices(path)["Close"]

    return {t: close_of(t) for t in canary_cfg["tickers"]}, close_of(canary_cfg["vix_ticker"])


def load_vix3m_close(config: dict, processed_dir: Path) -> pd.Series | None:
    """정제된 VIX3M 종가를 읽는다. 설정에 vix3m_ticker가 없거나(null) 기간구조 경보를 끄면 None."""
    canary_cfg = config["canary"]
    ticker = canary_cfg.get("vix3m_ticker")
    if ticker is None or canary_cfg.get("vix_term_threshold") is None:
        return None
    path = storage.find_path(processed_dir, ticker)
    if path is None:
        raise FileNotFoundError(f"[{ticker}] 정제 파일이 없습니다. 먼저 collect, clean 단계를 실행하세요.")
    return storage.load_prices(path)["Close"]


def term_kwargs(config: dict, processed_dir: Path) -> dict:
    """compute_risk_signals에 넘길 VIX 기간구조 인자 (설정에서 읽음)."""
    canary_cfg = config["canary"]
    return {
        "vix3m_close": load_vix3m_close(config, processed_dir),
        "term_threshold": canary_cfg.get("vix_term_threshold"),
        "term_level": canary_cfg.get("vix_term_level", 1),
    }


def load_risk_signals(config: dict, processed_dir: Path) -> pd.DataFrame:
    """정제된 카나리아·VIX 파일을 읽어 설정대로 위험 신호를 계산한다."""
    canary_cfg = config["canary"]
    canary_closes, vix_close = load_canary_closes(config, processed_dir)
    return compute_risk_signals(
        canary_closes, vix_close, canary_cfg["vix_threshold"], canary_cfg["method"],
        **term_kwargs(config, processed_dir),
    )
