"""시세 파일 이름 규칙과 읽기/쓰기 도우미.

파일 이름 규칙: <티커>_<종목명>.csv
    예: 005930_삼성전자.csv, SPY_State_Street_SPDR_SP_500_ETF_T.csv, ^VIX_CBOE_Volatility_Index.csv

- 티커는 그대로 쓴다. ('^VIX' → '^VIX_CBOE_Volatility_Index.csv')
  파일 이름의 첫 '_' 앞부분이 곧 티커라서, 파일만 보고도 어떤 티커인지 알 수 있다.
- 종목명을 모르면 '<티커>.csv'로 저장한다.
- 원본(data/raw)과 정제 데이터(data/processed) 모두 같은 규칙을 쓴다.
"""

import re
from pathlib import Path

import pandas as pd

MAX_NAME_LENGTH = 40  # 종목명이 너무 길면 자른다


def safe_ticker(ticker: str) -> str:
    """파일 이름에 쓸 수 있는 티커로 바꾼다.

    티커에 쓰이는 기호(^ . - =)는 파일 이름에도 쓸 수 있어 그대로 둔다. 예: '^VIX', 'BRK-B', 'KRW=X'
    """
    return re.sub(r"[^A-Za-z0-9.^=-]", "", ticker)


def ticker_from_path(path: Path) -> str:
    """파일 이름에서 티커를 꺼낸다. '005930_삼성전자.csv' → '005930'"""
    return path.stem.split("_", 1)[0]


def list_tickers(directory: Path) -> list[str]:
    """폴더에 저장된 모든 시세 파일의 티커 목록."""
    return [ticker_from_path(p) for p in sorted(directory.glob("*.csv"))]


def safe_name(name: str) -> str:
    """종목명을 파일 이름에 쓸 수 있게 바꾼다. 공백은 '_'로, 특수기호는 제거한다.

    예: 'SPDR S&P 500 ETF' → 'SPDR_SP_500_ETF', '삼성전자' → '삼성전자'
    """
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^\w.-]", "", name)  # \w는 한글도 포함한다
    name = re.sub(r"_+", "_", name)  # 기호를 지우며 생긴 '__'를 하나로
    return name[:MAX_NAME_LENGTH].strip("_")


def build_path(directory: Path, ticker: str, name: str | None) -> Path:
    """티커와 종목명으로 저장할 파일 경로를 만든다."""
    stem = safe_ticker(ticker)
    if name and safe_name(name):
        stem = f"{stem}_{safe_name(name)}"
    return directory / f"{stem}.csv"


def find_path(directory: Path, ticker: str) -> Path | None:
    """이미 저장된 티커의 파일을 찾는다. 없으면 None.

    '<티커>_<종목명>.csv'와 '<티커>.csv' 둘 다 찾는다.
    """
    stem = safe_ticker(ticker)
    candidates = sorted(directory.glob(f"{stem}_*.csv")) + [directory / f"{stem}.csv"]
    for path in candidates:
        if path.exists():
            return path
    return None


def name_from_path(path: Path, ticker: str) -> str | None:
    """파일 이름에서 종목명 부분을 꺼낸다. 'SPY_SPDR.csv' → 'SPDR', 'SPY.csv' → None"""
    prefix = f"{safe_ticker(ticker)}_"
    return path.stem[len(prefix):] if path.stem.startswith(prefix) else None


def load_prices(path: Path) -> pd.DataFrame:
    """저장된 시세 CSV를 Date 인덱스의 DataFrame으로 읽는다."""
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def save_prices(df: pd.DataFrame, directory: Path, ticker: str, name: str | None) -> Path:
    """시세를 규칙에 맞는 이름으로 저장한다.

    예전 이름의 파일(예: 종목명 없이 저장된 'SPY.csv')이 있으면 새 이름으로 바꾸고 예전 파일은 지운다.
    """
    directory.mkdir(parents=True, exist_ok=True)
    old_path = find_path(directory, ticker)
    new_path = build_path(directory, ticker, name)

    df.to_csv(new_path)
    if old_path is not None and old_path != new_path:
        old_path.unlink()
    return new_path
