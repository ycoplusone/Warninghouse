# Canary Dagger

카나리아 자산(신흥국 주식, 하이일드 채권)과 VIX로 시장 위험을 미리 감지해 주식 비중을 조절하는 퀀트 투자 연구 프로젝트입니다.

> 교육·연구 목적의 프로젝트입니다. 투자 권유가 아니며, 과거 성과가 미래 수익을 보장하지 않습니다.

---

## 전체 흐름

데이터를 **받고 → 다듬고 → 지표를 붙이고 → 과거로 시험**합니다. 각 단계는 앞 단계가 만든 파일을 읽어 다음 폴더에 저장합니다.

```
 인터넷(야후·네이버)
        │  ① collect      받기
        ▼
   data/raw/              원본 시세
        │  ② clean        다듬기
        ▼
   data/processed/        정제 시세 + 일별 수익률
        │  ③ indicators   지표 계산
        ▼
   data/features/         종목별 지표 + 위험 단계
   data/signals/          시장 전체 위험 신호
        │  ④ backtest     과거로 시험
        ▼
   reports/               전략 성과표
```

---

## 0. 준비

### 처음 한 번 (새 PC에서 시작할 때)

```bash
cd ~/canary_dagger
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 터미널을 열 때마다

```bash
cd ~/canary_dagger
source .venv/bin/activate
```

- **역할:** 이 프로젝트 전용 Python 환경을 켭니다. 켜지 않으면 `pandas`가 없다는 오류가 납니다.
- **결과:** 프롬프트 앞에 `(.venv)`가 붙습니다.

---

## ① 데이터 수집 — `collect`

```bash
python main.py --step collect
```

- **역할:** 설정 파일의 종목과 이전에 추가한 종목의 일별 시가·고가·저가·종가·거래량을 받습니다.
  - 해외 종목(SPY, ^VIX 등)은 yfinance, 국내 6자리 종목코드(005930 등)는 FinanceDataReader(네이버)에서 받습니다.
  - 이미 받은 종목은 새 날짜만 이어 받습니다. 배당·분할로 과거 가격이 바뀌었으면 처음부터 다시 받습니다.
- **저장:** `data/raw/<티커>_<종목명>.csv` (예: `data/raw/005930_삼성전자.csv`)
- **화면 예:**
  ```
  [SPY] 이어 받기 (+1행) — 총 4227행 (2010-01-04 ~ 2026-10-21)
  ```
- **새 종목 추가:**
  ```bash
  python main.py --step collect --ticker 005930 AAPL
  ```
  한 번 추가한 종목은 다음부터 `--ticker` 없이 실행해도 함께 업데이트됩니다.

**알아둘 점**
- 국내 종목은 네이버가 최근 약 3,000거래일(12년)까지만 제공해서 2014년경부터 받아집니다. 한 번 받은 뒤에는 이어 받기로 계속 쌓입니다.
- 국내 종가에는 배당이 반영되지 않습니다. 해외 종가는 배당·분할이 반영된 수정 종가입니다.

---

## ② 데이터 정제 — `clean`

```bash
python main.py --step clean
```

- **역할:** 원본의 잘못된 값을 고치고 하루 수익률을 계산합니다.

  | 규칙 | 예 |
  |---|---|
  | 중복 날짜 제거 | |
  | 장이 끝나지 않은 오늘 행 제거 | 미국 장 시작 전 VIX 값 |
  | 0 이하 가격을 빈 값으로 바꾼 뒤, 거래정지일은 종가로 채움 | 삼성전자 액면분할 기간 |
  | 하루 ±40% 넘게 튀었다가 다음 날 되돌아온 값은 오류로 보고 제거 | |
  | 중간의 빈 값은 직전 가격으로 채움 (최대 5일), 끝의 빈 행은 제거 | |
  | 일별 수익률 `Return` 컬럼 추가 | |

- **저장:** `data/processed/` (파일 이름 같음)
- **화면 예:**
  ```
  [005930] 0 이하 가격 3행 수정
  [^VIX] 미확정 오늘 1행 제거
  ```
- **결과 읽는 법:** `Return 0.0169` = 전날보다 1.69% 상승. 음수면 하락입니다.

---

## ③ 데이터 분석 — `indicators`

```bash
python main.py --step indicators
```

- **역할:** 두 가지를 계산합니다.
  1. **종목별 기술 지표**

     | 지표 | 뜻 |
     |---|---|
     | SMA 50 / 200 | 최근 50일·200일 종가 평균. 가격이 200일선 위면 상승 추세 |
     | MACD | 단기·장기 평균의 차이. 시그널선 위면 상승 힘이 강해지는 중 |
     | RSI | 0~100. 70 이상 과매수, 30 이하 과매도 |
     | 볼린저 밴드 %B | 0 = 하단, 0.5 = 중심, 1 = 상단 |

  2. **시장 전체 위험 단계 (카나리아 단검)**
     - 카나리아 자산(EEM, HYG)의 모멘텀 점수: 1·3·6·12개월 수익률을 최근일수록 크게 가중한 값. 0보다 작으면 하락 추세
     - **0 정상:** 카나리아 모두 상승 추세
     - **1 주의:** 카나리아 일부 하락 추세
     - **2 위험:** 카나리아 모두 하락 추세, 또는 VIX가 30 초과

- **저장:**
  - `data/features/` — 종목별 지표 + 위험 단계
  - `data/signals/risk_signals.csv` — 시장 전체 위험 신호
- **화면 예:**
  ```
  시장 위험 신호 (2026-10-20): 위험 — 카나리아 모멘텀 [EEM -0.207, HYG -0.082], VIX 17.3
  [SPY] 200일선 위(상승 추세) | 골든크로스 상태 | MACD 하락 흐름 | RSI 50(중립) | 시장 위험 위험
  ```

**알아둘 점**
- 한국 종목에는 '전날' 미국 위험 신호가 붙습니다. 한국 장이 끝나는 15:30에는 미국의 전날 종가까지만 알 수 있기 때문입니다.

---

## 오늘의 시장 상태 — `report`

```bash
python main.py --step report
explorer.exe "$(wslpath -w reports/market_today.html)"   # WSL에서 브라우저로 열기
```

- **역할:** ③의 결과를 숫자보다 문장 위주로 정리한 한 장짜리 HTML 리포트를 만듭니다.
  1. 오늘의 경보 단계(정상/주의/위험)와 그 의미
  2. 왜 그렇게 판단했나: 신흥국 주식·하이일드 채권·VIX 각각의 상태와 최근 1년 흐름
  3. 최근 1년 경보 기록 (날짜별 색 띠)
  4. 이 전략이라면 지금 어떤 비중일지 (미국·한국)
  5. 관심 종목 한눈에: 추세, 과열 여부, 최근 1개월 수익률
  6. 이 신호를 얼마나 믿어도 되나: 백테스트 결과 요약
- **저장:** `reports/market_today.html` (항상 최신), `reports/daily/market_<기준일>.html` (날짜별 사본)
- 그래프에 마우스를 올리면 날짜와 값이 보입니다. 다크 모드와 모바일 화면도 지원합니다.
- **매일 볼 때:** `python main.py --step collect && python main.py --step clean && python main.py --step indicators && python main.py --step report`

---

## ④ 백테스트 — `backtest`

```bash
python main.py --step backtest
```

- **역할:** 세 전략으로 투자했다면 어땠을지 미국·한국 시장별로 계산합니다. 수수료와 슬리피지를 반영합니다.
  - 미국: 2011년~ (설정 `universe`: SPY, QQQ / SHY, IEF)
  - 한국: 2014년~ (설정 `universe_kr`: KODEX 200 / 국고채 3년, 10년). 위험 신호는 미국 카나리아의 '전날' 신호를 씁니다.

  | 전략 | 내용 |
  |---|---|
  | SPY 보유 | SPY 하나를 사서 계속 보유 (비교 기준) |
  | 공격자산 보유 (방어 없음) | SPY·QQQ를 반씩, 매달 비중 재조정 |
  | 카나리아 단검 | 위험 단계에 따라 주식 비중 조절: 정상 100% / 주의 50% / 위험 0%. 나머지는 국채(SHY, IEF) |

  매월 마지막 거래일의 신호를 보고 다음 거래일 종가에 매매합니다. 미래 정보를 쓰지 않습니다.

- **저장:**
  - `reports/backtest_summary.csv` — 시장·전략별 성과 요약
  - `reports/backtest_equity_us.csv`, `reports/backtest_equity_kr.csv` — 날짜별 자산 가치
- **결과 읽는 법:**

  | 지표 | 뜻 |
  |---|---|
  | CAGR | 연평균 수익률. 1년에 평균 몇 %씩 불어났는지 |
  | MDD | 최대 낙폭. 최악의 순간에 고점보다 몇 % 떨어져 있었는지 |
  | 변동성 | 수익률이 얼마나 출렁이는지 |
  | 샤프 | CAGR ÷ 변동성. 클수록 좋음 |
  | 칼마 | CAGR ÷ MDD. 클수록 좋음 |

**알아둘 점**
- 백테스트 대상은 설정 파일의 `universe`, `universe_kr` 종목뿐입니다. `--ticker`로 추가한 종목은 들어가지 않습니다.
- 환율과 세금은 반영하지 않습니다. 한국 종가는 배당이 빠져 있어 수익률이 실제보다 조금 낮게 나옵니다.

---

## ⑤ 전략 개선 — `optimize`

```bash
python main.py --step optimize
```

- **역할:** 설정 값 조합을 모두 백테스트해서 가장 나은 조합을 고르고, 처음 보는 기간에서도 통하는지 검증합니다.

  | 바꿔 보는 항목 | 후보 |
  |---|---|
  | 방어 자산 | SHY+IEF / SHY / IEF / BIL / GLD / UUP / SHY+GLD / BIL+GLD |
  | '주의' 단계 주식 비중 | 50% / 75% / 100% |
  | VIX 기준값 | 25 / 30 / 35 / 사용 안 함 |
  | 카나리아 판단 방식 | momentum(모멘텀 점수) / sma200(200일선 하회) |

  모든 조합은 8 × 3 × 4 × 2 = 192가지입니다.

- **과최적화 막기:** 기간을 둘로 나눕니다.
  ```
  2011 ─────── 2018 │ 2019 ─────── 지금
    학습 기간: 조합 고르기  │  검증 기간: 고른 조합 시험
  ```
  - 조합은 **학습 기간 성적으로만** 고릅니다.
  - 기준은 칼마(CAGR ÷ MDD)입니다. 1등만 보지 않고 이웃 값(주의 비중·VIX가 한 칸 다른 조합)까지 평균 낸 '주변 칼마'로 고릅니다. 이웃에서 성적이 급락하는 '뾰족한 1등'은 우연일 가능성이 높기 때문입니다.
- **저장:**
  - `reports/optimize_results.csv` — 72가지 전체 결과
  - `reports/optimize_summary.csv` — 선택 조합 vs 방어 없음 vs 현재 설정 검증 결과
- **결과 읽는 법:** 마지막 '결론' 줄을 봅니다.
  - 선택 조합이 검증 기간에서 '방어 없음'보다 칼마가 높아야 진짜 개선입니다.
  - 학습 기간에서만 좋고 검증 기간에서 나빠지면 과거에만 맞은 값이므로 반영하지 않습니다.
- 설정 반영은 자동으로 하지 않습니다. 결과를 보고 `config/settings.yaml`을 직접 고칩니다.
- 후보 값과 기간 나누기는 `config/settings.yaml`의 `optimize`에서 바꿀 수 있습니다.

---

## ⑥ 그래프 — `charts`

```bash
python main.py --step charts
```

- **역할:** ④·⑤ 결과를 그림으로 저장합니다. ④ backtest와 ⑤ optimize를 먼저 실행해야 합니다.

  | 파일 | 내용 |
  |---|---|
  | `reports/chart_backtest_us.png` | 미국: 전략별 자산 곡선(위)과 낙폭(아래). 회색 음영은 '위험' 신호 구간 |
  | `reports/chart_backtest_kr.png` | 한국: 같은 형식 |
  | `reports/chart_defensive.png` | 방어 자산별 칼마가 학습 → 검증 기간에 어떻게 바뀌었는지. 선이 엇갈릴수록 과거 순위가 미래에 통하지 않았다는 뜻 |

- **결과 읽는 법:** 자산 곡선은 로그 눈금입니다. 세로축 한 칸이 같은 '배수'를 뜻해서, 오래 복리로 불어난 곡선을 공정하게 비교할 수 있습니다.
- 한글 폰트는 Windows의 맑은 고딕(WSL에서 `/mnt/c/Windows/Fonts`)이나 나눔고딕을 찾아 씁니다.

---

## 자주 쓰는 실행 방법

| 하고 싶은 일 | 명령어 |
|---|---|
| 전체 실행 (①~⑥ + 오늘의 리포트) | `python main.py` |
| 오늘의 시장 상태만 보기 | `python main.py --step report` → `reports/market_today.html` 열기 |
| 새 종목 분석 | `python main.py --step collect --ticker 005930`<br>`python main.py --step clean --ticker 005930`<br>`python main.py --step indicators --ticker 005930` |
| 설정만 바꾸고 다시 시험 | `config/settings.yaml` 수정 → `python main.py --step backtest` |

- 한 단계만 다시 돌리면 그 뒤 단계의 파일은 예전 상태로 남습니다. 예를 들어 ②를 다시 돌렸으면 ③과 ④도 다시 돌려야 결과에 반영됩니다.
- 매일 자동 실행이 필요하면 cron(리눅스) 또는 작업 스케줄러(Windows)로 `python main.py`를 등록합니다.

---

## 설정 파일 (`config/settings.yaml`)

| 항목 | 내용 |
|---|---|
| `data` | 수집 시작일, 데이터 폴더 위치 |
| `universe` | 미국 공격 자산(SPY, QQQ), 방어 자산(SHY, IEF), 벤치마크(SPY) |
| `universe_kr` | 한국 공격 자산(KODEX 200), 방어 자산(국고채 3년·10년), 벤치마크 |
| `canary` | 카나리아 자산(EEM, HYG), VIX 기준값(30), 판단 방식(momentum) |
| `cleaning` | 이상치 기준(40%), 빈 값 채우기 최대 일수(5일) |
| `indicators` | SMA, MACD, RSI, 볼린저 밴드 기간 |
| `backtest` | 초기 자금, 수수료, 슬리피지, 리밸런싱 주기, 위험 단계별 주식 비중 |
| `optimize` | 학습/검증 기간 경계, 바꿔 볼 후보 값 |

---

## 폴더 구조

```
canary_dagger/
├── main.py                  # 실행 시작점
├── config/settings.yaml     # 설정
├── src/
│   ├── config.py            # 설정 파일 읽기
│   ├── storage.py           # 파일 이름 규칙·읽기/쓰기
│   ├── collect.py           # ① 데이터 수집
│   ├── clean.py             # ② 데이터 정제
│   ├── indicators.py        # ③ 지표 계산
│   ├── risk.py              # ③ 카나리아 단검 위험 신호
│   ├── daily_report.py      # 오늘의 시장 상태 리포트
│   ├── backtest.py          # ④ 백테스트
│   ├── optimize.py          # ⑤ 전략 개선
│   └── charts.py            # ⑥ 그래프
├── data/
│   ├── raw/                 # ① 원본 시세
│   ├── processed/           # ② 정제 시세
│   ├── features/            # ③ 지표 + 위험 단계
│   └── signals/             # ③ 시장 위험 신호
├── reports/                 # ④⑤ 결과표, ⑥ 그래프
└── notebooks/               # 실험용 노트북
```

파일 이름 규칙: `<티커>_<종목명>.csv` (예: `005930_삼성전자.csv`, `^VIX_CBOE_Volatility_Index.csv`). 종목 하나당 파일 하나이며, 카나리아·VIX 같은 공용 데이터는 모든 종목 분석에서 함께 씁니다.
