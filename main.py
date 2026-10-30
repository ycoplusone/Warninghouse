"""Canary Dagger 파이프라인 실행 진입점.

사용 예:
    python main.py                  # 전체 파이프라인 실행
    python main.py --step collect   # 특정 단계만 실행
    python main.py --step collect --ticker 005930 AAPL   # 특정 종목만 실행 (새 종목 추가)
"""

import argparse
import logging

from src import backtest, charts, clean, collect, daily_report, indicators, optimize
from src.config import load_config

# 파이프라인 단계 실행 순서
STEPS = ["collect", "clean", "indicators", "report", "backtest", "optimize", "charts"]

# 단계 이름 → 실행할 모듈 (구현된 단계만 등록)
STEP_MODULES = {
    "collect": collect,
    "clean": clean,
    "indicators": indicators,
    "report": daily_report,
    "backtest": backtest,
    "optimize": optimize,
    "charts": charts,
}


def parse_args() -> argparse.Namespace:
    """명령줄 인자를 읽는다."""
    parser = argparse.ArgumentParser(description="Canary Dagger 퀀트 파이프라인")
    parser.add_argument(
        "--step",
        choices=STEPS,
        help="실행할 단계 (생략하면 전체 실행)",
    )
    parser.add_argument(
        "--ticker",
        nargs="+",
        help="처리할 티커/종목코드 (여러 개 가능). 생략하면 설정 파일 + 이미 받아 둔 모든 종목",
    )
    return parser.parse_args()


def run_step(step: str, config: dict, tickers: list[str] | None) -> None:
    """단계 이름에 해당하는 모듈의 run()을 실행한다."""
    module = STEP_MODULES.get(step)
    if module is None:
        logging.warning("단계 '%s'는 아직 구현되지 않았습니다. 건너뜁니다.", step)
        return
    logging.info("===== 단계 실행: %s =====", step)
    module.run(config, tickers)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    config = load_config()
    steps = [args.step] if args.step else STEPS
    for step in steps:
        run_step(step, config, args.ticker)


if __name__ == "__main__":
    main()
