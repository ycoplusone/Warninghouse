"""② 데이터 정제 — 결측치·이상치 처리, 수익률 계산 후 data/processed/에 저장

data/raw/의 원본을 읽어 아래 순서로 정리한 뒤, 같은 파일 이름으로 data/processed/에 저장한다.

1. 중복 날짜 제거 및 날짜순 정렬
2. 미확정 데이터 제거   : 아직 장이 끝나지 않은 '오늘' 행은 종가가 확정되지 않았으므로 뺀다
3. 잘못된 가격 처리     : 0 이하 가격은 있을 수 없으므로 빈 값(NaN)으로 바꾼다
4. 거래정지일 처리      : 거래가 없던 날(예: 액면분할 기간)은 시가·고가·저가를 종가로 채운다
5. 이상치(튀는 값) 처리 : 하루 급등(급락)했다가 다음 날 바로 되돌아오는 값은 데이터 오류로 보고 지운다
6. 결측치 처리          : 중간의 빈 값은 직전 가격으로 채우고, 맨 끝의 빈 행은 미확정이므로 뺀다
7. 수익률 계산          : Return = 오늘 종가 / 어제 종가 - 1

주의: 종목마다 거래일(휴장일)이 다르다. (예: 한국 추석 vs 미국 추수감사절)
      여기서는 종목별 원래 거래일을 그대로 두고, 여러 종목을 합치는 일은 ③ 분석 단계에서 한다.
"""

import logging
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src import storage
from src.collect import is_korean_ticker
from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["Open", "High", "Low", "Close"]

# 시장별 (시간대, 종가 확정 시각). 실제 장 마감보다 30분 여유를 둔다.
#   한국: 15:30 마감 → 16:00 이후 확정 / 미국: 16:00 마감(VIX는 16:15) → 16:30 이후 확정
MARKET_CLOSE = {
    "KR": (ZoneInfo("Asia/Seoul"), time(16, 0)),
    "US": (ZoneInfo("America/New_York"), time(16, 30)),
}


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """같은 날짜가 두 번 있으면 마지막 것만 남기고 날짜순으로 정렬한다."""
    return df[~df.index.duplicated(keep="last")].sort_index()


def drop_unconfirmed_today(df: pd.DataFrame, ticker: str, now: datetime | None = None) -> pd.DataFrame:
    """해당 시장 기준으로 장 마감 전이면 '오늘' 이후 행을 뺀다.

    Args:
        now: 현재 시각 (테스트용). None이면 실제 현재 시각.
    """
    tz, close_time = MARKET_CLOSE["KR" if is_korean_ticker(ticker) else "US"]
    local_now = (now or datetime.now(tz)).astimezone(tz)
    today = pd.Timestamp(local_now.date())

    if local_now.time() >= close_time:
        return df[df.index <= today]  # 오늘 장은 확정됨
    return df[df.index < today]


def fix_invalid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """0 이하의 가격을 빈 값(NaN)으로 바꾼다."""
    df = df.copy()
    df[PRICE_COLUMNS] = df[PRICE_COLUMNS].where(df[PRICE_COLUMNS] > 0)
    return df


def fill_halt_days(df: pd.DataFrame) -> pd.DataFrame:
    """종가는 있는데 시가·고가·저가가 비어 있는 날(거래정지일 등)은 종가로 채운다."""
    df = df.copy()
    for col in ["Open", "High", "Low"]:
        df[col] = df[col].fillna(df["Close"])
    return df


def find_spikes(close: pd.Series, threshold: float) -> pd.Series:
    """하루 튀었다가 다음 날 바로 되돌아온 날을 찾는다.

    오늘 수익률과 다음 날 수익률이 둘 다 threshold보다 크고 방향이 반대면 데이터 오류로 본다.
    예) 100 → 200 → 100 : 가운데 200은 잘못 기록된 값일 가능성이 높다.
    단순히 크게 움직인 날(예: VIX 급등)은 다음 날 되돌아오지 않으면 실제 움직임으로 보고 남긴다.

    Returns:
        이상치인 날짜가 True인 Series
    """
    ret = close.pct_change(fill_method=None)
    next_ret = ret.shift(-1)
    return (ret.abs() > threshold) & (next_ret.abs() > threshold) & (np.sign(ret) != np.sign(next_ret))


def fill_missing(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """빈 가격을 처리한다.

    - 맨 끝의 빈 행: 아직 확정되지 않은 데이터이므로 뺀다. (채우면 실제로 움직였어도 수익률 0%가 됨)
    - 중간의 빈 값: 직전 가격으로 채운다(forward fill). 최대 limit일 연속까지만 채운다.
    - 그래도 남은 빈 행(맨 앞, 또는 너무 길게 빈 구간)은 뺀다.
    """
    last_valid = df["Close"].last_valid_index()
    df = df.loc[:last_valid]
    df = df.copy()
    df[PRICE_COLUMNS] = df[PRICE_COLUMNS].ffill(limit=limit)
    return df.dropna(subset=["Close"])


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """일별 수익률(Return) 컬럼을 추가한다. 첫날은 전날이 없어 NaN이다."""
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    return df


def clean_prices(df: pd.DataFrame, ticker: str, outlier_threshold: float, ffill_limit: int) -> pd.DataFrame:
    """원본 시세를 정제하고 수익률을 붙여 돌려준다. 처리 내용은 로그로 남긴다."""
    n_raw = len(df)

    df = drop_duplicates(df)
    n_dup = n_raw - len(df)

    before = len(df)
    df = drop_unconfirmed_today(df, ticker)
    n_today = before - len(df)

    n_invalid = int((df[PRICE_COLUMNS] <= 0).any(axis=1).sum())
    df = fix_invalid_prices(df)
    df = fill_halt_days(df)

    spikes = find_spikes(df["Close"], outlier_threshold)
    df.loc[spikes, PRICE_COLUMNS] = np.nan

    n_missing = int(df["Close"].isna().sum())
    before = len(df)
    df = fill_missing(df, ffill_limit)
    n_dropped = before - len(df)

    df = add_returns(df)

    # 처리한 내용이 있을 때만 자세히 기록
    notes = []
    if n_dup:
        notes.append(f"중복 {n_dup}행 제거")
    if n_today:
        notes.append(f"미확정 오늘 {n_today}행 제거")
    if n_invalid:
        notes.append(f"0 이하 가격 {n_invalid}행 수정")
    if spikes.any():
        spike_dates = [d.date().isoformat() for d in spikes[spikes].index]
        notes.append(f"이상치 {len(spike_dates)}일 {spike_dates}")
    if n_missing:
        notes.append(f"빈 종가 {n_missing}개 (채움 {n_missing - n_dropped}, 제거 {n_dropped})")
    logger.info("[%s] %s", ticker, ", ".join(notes) if notes else "문제 없음")

    return df


def clean_ticker(ticker: str, raw_dir: Path, processed_dir: Path, outlier_threshold: float, ffill_limit: int) -> Path:
    """한 종목의 원본을 읽어 정제하고 저장 경로를 돌려준다."""
    raw_path = storage.find_path(raw_dir, ticker)
    if raw_path is None:
        raise FileNotFoundError(f"원본 파일이 없습니다. 먼저 수집하세요: python main.py --step collect --ticker {ticker}")

    df = clean_prices(storage.load_prices(raw_path), ticker, outlier_threshold, ffill_limit)
    name = storage.name_from_path(raw_path, ticker)
    path = storage.save_prices(df, processed_dir, ticker, name)
    logger.info("[%s] 총 %d행 (%s ~ %s) → %s", ticker, len(df), df.index[0].date(), df.index[-1].date(), path.name)
    return path


def run(config: dict, tickers: list[str] | None = None) -> dict[str, Path]:
    """파이프라인 ② 단계: 원본 시세를 정제해 data/processed/에 저장한다.

    Args:
        config: 설정 딕셔너리
        tickers: 정제할 티커 목록. None이면 data/raw/에 있는 모든 종목.
    """
    data_cfg = config["data"]
    clean_cfg = config["cleaning"]
    raw_dir = PROJECT_ROOT / data_cfg["raw_dir"]
    processed_dir = PROJECT_ROOT / data_cfg["processed_dir"]
    tickers = tickers or storage.list_tickers(raw_dir)

    logger.info("정제 시작: %d개 종목", len(tickers))
    saved: dict[str, Path] = {}
    for ticker in tickers:
        try:
            saved[ticker] = clean_ticker(
                ticker, raw_dir, processed_dir, clean_cfg["outlier_threshold"], clean_cfg["ffill_limit"],
            )
        except Exception as e:
            logger.error("[%s] 정제 실패: %s", ticker, e)
    return saved
