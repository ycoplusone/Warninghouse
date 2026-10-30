"""① 데이터 수집 — yfinance / FinanceDataReader로 시세를 받아 data/raw/에 저장

- 해외 티커(SPY, ^VIX 등)  → yfinance
- 국내 6자리 종목코드(069500 등) → FinanceDataReader (네이버 금융)

국내 데이터 주의사항:
  * 네이버는 한 번에 최근 3,000거래일(약 12년)까지만 준다. 그래서 2010년부터 요청해도 2014년경부터 받아진다.
    한 번 받아 둔 뒤에는 이어 받기로 계속 쌓이므로 앞으로는 기간이 줄지 않는다.
  * 야후(005930.KS)는 2010년부터 주지만 종가가 거래소 공식 종가와 자주 달라서(0.5~2%) 쓰지 않는다.
  * 국내 종가는 분할만 반영되고 배당은 반영되지 않는다. (해외 수정 종가와 달리 배당 수익이 빠진다)

저장 형식: data/raw/<티커>_<종목명>.csv  (파일 이름 규칙은 src/storage.py 참고)
컬럼: Date(인덱스), Open, High, Low, Close, Volume
  * Close는 배당·분할이 반영된 '수정 종가'다. 수익률 계산은 이 값을 기준으로 한다.

이어 받기:
  이미 받은 파일이 있으면 마지막 며칠치만 다시 받아 뒤에 붙인다.
  단, 그사이 배당·분할이 생겨 과거 수정 종가가 바뀌었다면 처음부터 다시 받는다.
  (수정 종가는 배당이 나올 때마다 과거 가격 전체가 다시 계산되기 때문)
"""

import logging
import re
import time
from functools import lru_cache
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
import yfinance as yf

from src import storage
from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# 이어 받을 때 기존 데이터와 겹치게 다시 받을 행 수.
# 겹치는 구간의 종가를 비교해 과거 가격이 바뀌었는지 확인하는 데 쓴다.
OVERLAP_ROWS = 5

# 겹치는 구간 종가 비교 허용 오차 (0.01%)
PRICE_TOLERANCE = 1e-4


def is_korean_ticker(ticker: str) -> bool:
    """국내 종목코드(숫자 6자리)인지 확인한다. 예: '005930'(삼성전자), '069500'(KODEX 200)"""
    return bool(re.fullmatch(r"\d{6}", ticker))


# ---------------------------------------------------------------------------
# 종목명 찾기
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _korean_name_table() -> dict[str, str]:
    """국내 주식 + ETF의 {종목코드: 종목명} 표. 한 번만 받아서 재사용한다."""
    stocks = fdr.StockListing("KRX")
    etfs = fdr.StockListing("ETF/KR")
    table = dict(zip(stocks["Code"], stocks["Name"]))
    table.update(zip(etfs["Symbol"], etfs["Name"]))
    return table


def lookup_name(ticker: str) -> str | None:
    """티커의 종목명을 찾는다. 찾지 못하면 None (파일 이름에 종목명 없이 저장된다)."""
    try:
        if is_korean_ticker(ticker):
            return _korean_name_table().get(ticker)
        return yf.Ticker(ticker).info.get("shortName")
    except Exception as e:
        logger.warning("[%s] 종목명 조회 실패: %s", ticker, e)
        return None


# ---------------------------------------------------------------------------
# 시세 다운로드
# ---------------------------------------------------------------------------

def _download_overseas(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """yfinance로 해외 시세를 받는다. auto_adjust=True면 Close가 수정 종가가 된다."""
    return yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,  # 컬럼을 ('Close','SPY') 대신 'Close' 한 단계로 받기
    )


def _download_korean(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """FinanceDataReader로 국내 시세를 받는다."""
    return fdr.DataReader(ticker, start, end)


def download_prices(
    ticker: str,
    start: str,
    end: str | None = None,
    retries: int = 3,
    wait_seconds: float = 2.0,
) -> pd.DataFrame:
    """한 종목의 일별 시세를 내려받는다.

    네트워크 오류나 일시적인 서버 문제에 대비해 실패하면 잠시 기다렸다 다시 시도한다.

    Args:
        ticker: 티커 또는 국내 6자리 종목코드
        start: 시작일 ('YYYY-MM-DD')
        end: 종료일 ('YYYY-MM-DD'), None이면 오늘까지
        retries: 최대 시도 횟수
        wait_seconds: 재시도 전 대기 시간(초). 시도할 때마다 2배로 늘린다.

    Returns:
        Date 인덱스와 Open/High/Low/Close/Volume 컬럼을 가진 DataFrame
    """
    downloader = _download_korean if is_korean_ticker(ticker) else _download_overseas

    for attempt in range(1, retries + 1):
        try:
            df = downloader(ticker, start, end)
            if df is None or df.empty:
                raise ValueError("받은 데이터가 비어 있음 (티커 또는 기간을 확인하세요)")

            # 필요한 컬럼만 남기고, 인덱스 이름을 통일한다.
            # (VIX 같은 지수는 거래량이 없을 수 있어 있는 컬럼만 고른다)
            df = df[[c for c in PRICE_COLUMNS if c in df.columns]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"
            return df.sort_index()

        except Exception as e:
            logger.warning("[%s] 다운로드 실패 (%d/%d): %s", ticker, attempt, retries, e)
            if attempt < retries:
                time.sleep(wait_seconds * 2 ** (attempt - 1))

    raise RuntimeError(f"[{ticker}] {retries}번 시도했지만 데이터를 받지 못했습니다.")


# ---------------------------------------------------------------------------
# 이어 받기
# ---------------------------------------------------------------------------

def _past_prices_unchanged(old: pd.DataFrame, new: pd.DataFrame) -> bool:
    """겹치는 날짜의 종가가 기존 파일과 같은지 확인한다.

    기존 파일의 마지막 행은 장중에 받은 '미확정' 값일 수 있어 비교에서 뺀다.
    """
    common = old.index[:-1].intersection(new.index)
    if common.empty:
        return False
    old_close = old.loc[common, "Close"]
    new_close = new.loc[common, "Close"]
    return bool(((old_close - new_close).abs() / old_close.abs() <= PRICE_TOLERANCE).all())


def update_prices(ticker: str, start: str, end: str | None, old: pd.DataFrame | None) -> tuple[pd.DataFrame, str]:
    """기존 데이터가 있으면 이어 받고, 없으면 처음부터 받는다.

    Returns:
        (전체 시세, 수집 방식 설명) — 수집 방식은 로그에 남기는 용도
    """
    if old is None or len(old) < OVERLAP_ROWS:
        return download_prices(ticker, start, end), "전체 다운로드"

    # 마지막 OVERLAP_ROWS행부터 다시 받는다 (겹치는 구간으로 과거 가격 변동 여부 확인)
    overlap_start = old.index[-OVERLAP_ROWS].strftime("%Y-%m-%d")
    new = download_prices(ticker, overlap_start, end)

    if not _past_prices_unchanged(old, new):
        # 배당·분할로 과거 수정 종가가 바뀌었다 → 이어 붙이면 가격이 어긋나므로 전부 다시 받는다
        return download_prices(ticker, start, end), "과거 가격 변경 감지 → 전체 다시 다운로드"

    # 기존 데이터 중 새로 받은 구간 이전 부분 + 새로 받은 데이터
    merged = pd.concat([old[old.index < new.index[0]], new])
    added = len(merged) - len(old)
    return merged, f"이어 받기 (+{added}행)"


def collect_ticker(ticker: str, start: str, end: str | None, raw_dir: Path) -> Path:
    """한 종목을 수집(또는 이어 받기)해 저장하고 저장 경로를 돌려준다."""
    existing_path = storage.find_path(raw_dir, ticker)
    old = storage.load_prices(existing_path) if existing_path else None

    df, mode = update_prices(ticker, start, end, old)

    # 종목명: 기존 파일 이름에 있으면 그대로 쓰고, 없을 때만 새로 조회한다
    name = storage.name_from_path(existing_path, ticker) if existing_path else None
    name = name or lookup_name(ticker)

    path = storage.save_prices(df, raw_dir, ticker, name)
    logger.info(
        "[%s] %s — 총 %d행 (%s ~ %s) → %s",
        ticker, mode, len(df), df.index[0].date(), df.index[-1].date(), path.name,
    )
    return path


def collect_tickers(tickers: list[str], start: str, end: str | None, raw_dir: Path) -> dict[str, Path]:
    """여러 종목을 차례로 받아 저장한다.

    한 종목이 실패해도 나머지는 계속 받는다. 실패한 종목은 로그로 알려준다.

    Returns:
        {티커: 저장된 파일 경로} — 성공한 종목만 들어 있다.
    """
    saved: dict[str, Path] = {}
    failed: list[str] = []

    for ticker in tickers:
        try:
            saved[ticker] = collect_ticker(ticker, start, end, raw_dir)
        except Exception as e:
            failed.append(ticker)
            logger.error("[%s] 수집 실패: %s", ticker, e)

    if failed:
        logger.error("수집 실패 종목: %s", ", ".join(failed))
    return saved


def get_all_tickers(config: dict, raw_dir: Path) -> list[str]:
    """수집할 티커 목록을 중복 없이 모은다.

    1. 설정 파일의 티커 (공격·방어·벤치마크·카나리아·VIX)
    2. 이미 data/raw/에 받아 둔 티커 (--ticker로 추가했던 종목도 매번 함께 업데이트된다)
    """
    universe = config["universe"]
    canary = config["canary"]
    tickers = (
        universe["offensive"]
        + universe["defensive"]
        + [universe["benchmark"]]
        + canary["tickers"]
        + [canary["vix_ticker"]]
        + storage.list_tickers(raw_dir)
    )
    return list(dict.fromkeys(tickers))  # 순서를 유지하면서 중복 제거


def run(config: dict, tickers: list[str] | None = None) -> dict[str, Path]:
    """파이프라인 ① 단계: 시세를 수집한다.

    Args:
        config: 설정 딕셔너리
        tickers: 수집할 티커 목록. None이면 설정 파일 + 이미 받아 둔 모든 종목을 업데이트한다.
    """
    data_cfg = config["data"]
    raw_dir = PROJECT_ROOT / data_cfg["raw_dir"]
    tickers = tickers or get_all_tickers(config, raw_dir)

    logger.info("수집 시작: %d개 종목 %s", len(tickers), tickers)
    return collect_tickers(tickers, data_cfg["start_date"], data_cfg["end_date"], raw_dir)
