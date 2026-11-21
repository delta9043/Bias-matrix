# Bias-Matrix Editing and Augmentation

논문 "Bias-Matrix Editing and Augmentation: Debiasing from Embeddings to Training Data"의 코드와 실험 결과임.

- 문제: LLM 임베딩 위에서 분류 모델을 학습하면, 학습 데이터에서 클래스(과제 라벨)와 그룹(보호 속성)이 함께 나타나는 정도가 그룹 간 재현율(TPR) 격차로 이어짐
- 조건부 편향 행렬 `B`: 클래스 안에서 그룹 평균이 클래스 평균과 다른 정도를 열로 쌓은 행렬 (`d × CG`). SVD의 왼쪽 특이벡터가 클래스 안의 그룹 차이가 큰 방향임
- **BME** (Bias-Matrix Editing, 편향 행렬 편집법): 임베딩에서 `B`의 상위 `k`개 특이방향 성분을 지움 (논문의 알고리즘 1)
- **BMA** (Bias-Matrix Augmentation, 편향 행렬 증강법): 임베딩은 그대로 두고, 같은 클래스의 다른 그룹 표본을 그룹 평균 차이만큼 옮긴 사본을 학습 데이터에 추가함 (논문의 알고리즘 2)
- 실험: Bias in Bios (직업 28종, 성별)와 공공데이터포털의 위해위험 소비자상담 내역 (품목 대분류 15종, 판매방식). 임베딩은 Qwen3-0.6B의 마지막 층 hidden state 평균 (`d = 1024`), 모델은 다항 로지스틱 회귀임

## 프로젝트 페이지와 데모

- `docs/`: 프로젝트 페이지 (정적 HTML, GitHub Pages의 `/docs` 배포용). 방법이 적용되는 과정을 보여 주는 애니메이션 (그림 1), 페이지 안에서 직접 그려지는 그림 2~4와 표 2~3, 브라우저에서 계산되는 데모 두 가지를 담음
  - 데모 1: 2차원 합성 데이터 (클래스 2개, 그룹 2개)에서 학습 데이터의 편향 강도를 바꾸고, 원본·BME·BMA의 결정 경계와 그룹별 재현율을 비교함
  - 데모 2: Bias in Bios의 의료 직업 7종 임베딩 (학습 표본 8,000건, 주성분 64개로 축소, 테스트 28,282건)으로 모델을 브라우저에서 학습하고, BME의 방향 수와 모델 고정 여부, BMA의 추가량을 바꿔 가며 직업별 재현율과 예시 약력의 예측을 비교함. 본문의 실험을 줄인 설정이라 수치는 표와 다름
- 로컬에서 보기: `docs/index.html`을 브라우저로 열거나 `python -m http.server --directory docs 8000` 실행 후 `http://localhost:8000` (수식과 글꼴은 CDN에서 받으므로 인터넷 연결이 필요함)
- 배포: 저장소 Settings → Pages → Branch `main`, 폴더 `/docs`
- `docs/assets/bm-core.js`: 데모의 계산 코어 (소프트맥스 회귀 L-BFGS, 편향 행렬, BME, 최소 증강 계획과 BMA). 그룹이 둘인 경우만 구현함
- `demo/`: 페이지에 들어가는 데이터를 만들고 확인하는 스크립트 (저장소 루트에서 실행)

| 명령 | 하는 일 |
|---|---|
| `python -m demo.build_page_data` | `results/`의 표, 본문 수치, 그림 2~4의 계열, 애니메이션의 합성 표본 → `docs/assets/page-data.js` (임베딩 필요 없음) |
| `python -m demo.build_demo_data` | 데모 2의 데이터 → `docs/assets/demo-data.js` (Bias in Bios 임베딩과 LLM 필요). 예시 약력은 `demo/examples.json`의 지어낸 문장이며 실존 인물의 약력은 넣지 않음 |
| `python -m demo.reference && node demo/check_core.js` | 데모 데이터에서 Python 라이브러리와 브라우저 계산 코어의 결과를 대조 (원본, BME, 최소 계획, BMA) |

## 구성

| 경로 | 내용 |
|---|---|
| `bias_matrix/` | 방법 라이브러리 (numpy, scipy만 사용) |
| `bias_matrix/stats.py` | 칸(클래스와 그룹의 조합)별 평균과 표본 수, 공분산, 상호정보량 |
| `bias_matrix/bme.py` | 조건부 편향 행렬, BME 편집 연산 (직교·사교, hard·soft, 상위 `k`개), `k`를 정하는 순차 순열검정 |
| `bias_matrix/bma.py` | 최소 증강 계획, 사본을 만들 원본 뽑기, 평균을 옮긴 사본, BMA |
| `experiments/common.py` | 경로, 임베딩과 결과 입출력, 그림 양식 |
| `experiments/data.py` | Bias in Bios 로드, 성별 단어 치환 (CDA용), 위해위험 상담 로드 |
| `experiments/embed.py` | Qwen3-0.6B 임베딩 추출 (층별 평균 풀링, 청크 단위로 이어서 진행) |
| `experiments/models.py` | 선형 모델 (다항 로지스틱 회귀, L2 정규화, torch L-BFGS) |
| `experiments/metrics.py` | 재현율 격차 지표 (RMS, bootstrap으로 보정한 GAP과 95% 구간, 소수·다수 그룹 재현율) |
| `experiments/baselines.py` | 비교 방법 (LEACE, INLP와 직업 보정 변형, 데이터 행렬 `X`의 상위 특이값 제거, reweighing) |
| `experiments/<실험>.py` | 실험 하나에 스크립트 하나 (아래 "실험과 본문의 대응") |
| `experiments/figures.py`, `experiments/tables.py` | 결과 JSON에서 본문의 그림 1~4, 표 2~3, 본문 수치를 만듦 |
| `tests/test_math.py` | 명제와 구현의 수치 검증 (합성 데이터) |
| `results/` | 본문에 실린 결과 (`*.json`), 그림 (`figures/`), 표와 본문 수치 (`tables.md`, `tables.json`) |
| `run_embed.sh`, `run_experiments.sh` | 전체 실행 스크립트 |
| `docs/` | 프로젝트 페이지와 데모 (위 "프로젝트 페이지와 데모") |
| `demo/` | 페이지의 데이터를 만들고 확인하는 스크립트 |

- 방법만 쓰려면 `bias_matrix/`만 있으면 됨 (아래 "라이브러리 사용 예"). `experiments/`와 `results/`는 본문의 실험을 다시 만들기 위한 것임

## 설치

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/xpu   # Intel GPU 빌드 (다른 환경은 그에 맞는 torch)
.venv/Scripts/python -m pip install transformers huggingface_hub numpy scipy pandas pyarrow matplotlib pytest concept-erasure
```

- 결과를 얻은 환경은 Python 3.14이며 패키지 버전은 `requirements.txt`에 있음
- 계산 장치는 Intel GPU(`xpu`), CUDA, CPU 순으로 고르며 각 스크립트의 `--device`로 바꿀 수 있음. 본문의 결과는 Intel Arc 내장 GPU에서 얻었고, CUDA 경로는 실행해 보지 못함
- `concept-erasure`는 LEACE 구현과 대조하는 테스트 하나에만 쓰이며, 없으면 그 테스트는 건너뜀

## 데이터

- Bias in Bios: `LabHC/bias_in_bios` (Hugging Face, MIT). 첫 실행 때 `data/bias_in_bios/`로 내려받음 (학습 257,478건, dev 39,642건, 테스트 99,069건)
- 임베딩 모델: `Qwen/Qwen3-0.6B` (Hugging Face, Apache-2.0). 첫 실행 때 Hugging Face 캐시로 내려받음
- 위해위험 소비자상담 내역: 공공데이터포털의 「공정거래위원회_소비자 민원학습데이터 위해위험 소비자상담 내역 데이터」 (https://www.data.go.kr/data/15098320/fileData.do, 이용허락범위 제한 없음). 내려받은 CSV를 `data/consumer/hazard_15098320.csv`로 둠 (CP949, 129,759행 가운데 열 수가 맞지 않는 64행은 건너뜀)
  - 판매방식코드 10을 일반 판매, 24와 27을 비대면 판매로 묶음. 코드표를 찾지 못해 24와 27의 뜻(인터넷 판매, 홈쇼핑 판매)은 상담 본문의 단어로 추정함 (24의 36%에 인터넷·쇼핑몰·온라인, 27의 78%에 홈쇼핑이 나오고, 10에서는 각각 4%, 0.5%)
  - 클래스는 품목코드 앞 두 자리(대분류)이며, 1,000건 이상이고 두 판매방식에 각각 100건 이상 있는 15종만 씀 (학습 83,641건, 테스트 20,911건, 8:2 무작위 분할)
  - 품목 대분류의 이름은 같은 포털의 품목코드표 (https://www.data.go.kr/data/15098317/fileData.do)에서 확인할 수 있음. 파일을 `data/consumer/itemcode_15098317.csv`로 두면 되며, 이 저장소의 코드는 이 파일을 읽지 않음
- 데이터 폴더는 환경 변수 `BIAS_MATRIX_DATA`로 바꿀 수 있음 (기본값 `./data`). 추출한 임베딩은 `data/emb/Qwen3-0.6B/` 아래에 약 6GB를 차지함 (층 7, 14, 21, 28을 float16으로 저장)

## 실행

저장소 루트에서 실행함 (Windows는 Git Bash 기준).

```bash
export PYTHON=.venv/Scripts/python     # 쓰지 않으면 PATH의 python
bash run_embed.sh                      # 임베딩 추출 (중단한 뒤 다시 실행하면 끝난 청크는 건너뜀)
bash run_experiments.sh                # 테스트 -> 층과 L2 선택 -> 실험 -> 그림과 표
```

- 실험은 하나씩 따로 실행할 수 있음: `python -m experiments.removal`, `python -m experiments.dose --data hazard` 등
- 실험을 다시 돌리면 `results/`의 JSON을 덮어씀. 실린 결과를 그대로 두려면 `BIAS_MATRIX_RESULTS=<폴더>`로 다른 곳에 쓰게 함
- 그림과 표만 다시 만들 때는 임베딩이 필요 없음: `python -m experiments.figures`, `python -m experiments.tables`
- 임베딩 추출은 Intel Arc 내장 GPU(fp16)에서 초당 약 100문장, CPU에서 초당 약 6문장임

## 라이브러리 사용 예

```python
import numpy as np
from bias_matrix import bma, bme, select_rank

# X: (n, d) 임베딩, y: 클래스 0..C-1, z: 그룹 0..G-1 (정수 배열)
eraser = bme(X_train, y_train, z_train)            # BME를 학습 데이터에서 추정 (모든 방향, 사교 편집)
X_train_e, X_test_e = eraser(X_train), eraser(X_test)   # 학습·테스트 임베딩에 같은 변환 적용

k, s, thr = select_rank(X_train, y_train, z_train, n_perm=200, rng=0)   # 순차 순열검정으로 방향의 수 k 선택
eraser_k = bme(X_train, y_train, z_train, k=k, oblique=False)           # 알고리즘 1: 상위 k개 방향의 직교 편집

X_new, y_new, z_new = bma(X_train, y_train, z_train, rng=0)   # BMA: 최소 계획의 칸에 넣을 사본
X_aug, y_aug = np.concatenate([X_train, X_new]), np.concatenate([y_train, y_new])   # 이 집합으로 모델만 다시 학습
```

- `bme(..., shrink=0.25)`는 soft editing (특이값을 0.25배로 줄임), `oblique=True`(기본값)는 공분산으로 백색화한 공간에서 편집하는 사교 편집임
- 통계량을 따로 구해 여러 편집을 만들 때는 `conditional_bias_matrix`, `covariance`, `bme_eraser(B, mean, cov, k, shrink)`를 씀
- BMA의 단계를 나누어 쓸 때는 `cell_stats`, `minimal_plan`, `plan_sources`, `mean_shift`를 씀

## 실험과 본문의 대응

| 명령 | 결과 파일 | 본문 |
|---|---|---|
| `python -m experiments.select_layer` | `select.json` | 임베딩 층과 L2 세기 선택 (dev 정확도 기준, 층 28과 L2 0.001) |
| `python -m experiments.spectrum` | `spectrum.json` | 그림 1의 4단계 (특이값과 순열검정 임계값), 5.2절의 선택된 방향의 수 |
| `python -m experiments.removal` | `removal.json` | 표 2 위쪽 (임베딩 편집), 그림 2b의 LEACE·INLP, 그림 2c |
| `python -m experiments.frozen` | `frozen.json` | 표 2의 "BME (Ours, 모델 고정)" |
| `python -m experiments.augmentation` | `augmentation.json` | 표 2 아래쪽 (reweighing, CDA, BMA), 그림 2b의 점, 그림 3c |
| `python -m experiments.dose --data bios` | `dose_bios.json` | 그림 2a, 그림 3a·3b |
| `python -m experiments.tradeoff --data bios` | `tradeoff_bios.json` | 그림 2b의 BME 선 두 개 |
| `python -m experiments.hazard` | `hazard.json` | 표 3 위쪽 (그룹 2개), 그림 4b의 점, 그림 4c |
| `python -m experiments.multigroup` | `multigroup.json` | 표 3 아래쪽 (그룹 3개) |
| `python -m experiments.rare_cells` | `rare_cells.json` | 5.4절의 드문 품목 재현율 |
| `python -m experiments.dose --data hazard` | `dose_hazard.json` | 그림 4a |
| `python -m experiments.tradeoff --data hazard --inlp` | `tradeoff_hazard.json` | 그림 4b의 BME 선과 INLP |
| `python -m experiments.figures` | `figures/fig1_overview.png` ~ `fig4_korea.png` | 그림 1~4 |
| `python -m experiments.tables` | `tables.md`, `tables.json` | 표 2~3, 본문 수치 |

- 결과 행의 이름: `none` 원본 모델, `bme` BME (사교 편집, 모든 방향, 재학습), `bme-orth-k` 직교 편집으로 상위 `k`개 제거, `bme-soft-0.25` soft editing, `bme-frozen` 모델을 고정하고 임베딩만 편집, `bme-uniform` 편향 행렬의 열에 표본 수 가중치를 주지 않고 방향의 순서를 정한 경우 (그림의 "클래스 균등"), `X-top-k` 데이터 행렬의 상위 `k`개 제거, `leace`, `inlp-t` (반복 `t`회), `inlp-class-t` (직업 보정 변형), `reweighing`, `cda-all`·`cda-partial` (모든 문장 또는 BMA와 같은 칸에만 성별 단어를 바꾼 문장 추가), `bma`, `bma-all` (모든 표본의 사본 추가)
- 지표: `acc` 정확도, `gap_rms` 클래스별 재현율 격차의 RMS, `gap_rms_bc`와 `gap_rms_ci` bootstrap으로 보정한 값(본문의 보정 GAP)과 95% 구간, `tpr_minority`·`tpr_majority` 각 클래스에서 학습 데이터가 가장 적은·많은 그룹의 재현율 평균

## 재현성

- 난수 시드는 스크립트 안에 고정함. 다만 GPU에서 학습한 모델은 실행마다 비트 단위로 같지 않아, 다시 실행하면 테스트 예측 일부가 달라짐 (편향 강도 실험의 한 설정에서 같은 학습을 되풀이했을 때 99,069개 가운데 45~72개)
- `results/`의 JSON은 본문의 수치를 낸 실행의 결과이며, 본문에 쓰지 않는 행과 항목만 덜어 냄. 따라서 `python -m experiments.figures`와 `python -m experiments.tables`는 본문의 그림 (픽셀 단위로 같음), 표, 수치를 그대로 다시 만듦
- 이 저장소의 스크립트를 같은 임베딩으로 다시 실행해 `results/`와 비교한 차이는 아래와 같음 (2026-11-21, Intel Arc 내장 GPU. 편향 강도 실험 `dose`는 5회 반복 가운데 첫 번째만 다시 실행함)

| 실험 | 정확도 차이의 최댓값 (%p) | 보정 GAP 차이의 최댓값 | 실행 시간 |
|---|---|---|---|
| `select_layer` | 0.028 (dev 정확도) | -- | 5분 |
| `spectrum` | -- | -- (선택된 `k`가 같고 특이값과 임계값의 차이는 1e-15 이하) | 1분 미만 |
| `removal` | 0.019 | 0.0015 | 17분 |
| `frozen` | 0.008 | 0.0004 | 1분 미만 |
| `augmentation` | 0.005 | 0.0007 | 4분 |
| `dose --data bios` | 0.016 | 0.0056 | 14분 (1회 반복) |
| `tradeoff --data bios` | 0.013 | 0.0012 | 9분 |
| `hazard` | 0.010 | 0.0002 | 1분 미만 |
| `multigroup` | 0.008 | 0.0006 | 1분 미만 |
| `rare_cells` | 0.014 | -- (드문 칸의 재현율은 같음) | 1분 미만 |
| `dose --data hazard` | 0.019 | 0.0007 | 5분 (1회 반복) |
| `tradeoff --data hazard --inlp` | 0.019 | 0.0007 | 2분 |

- 보정 GAP의 차이는 대부분 0.001 이내임. Bias in Bios의 편향 강도 실험에서는 다시 실행한 30개 행 가운데 2개가 더 크게 달랐음 (BME의 α=0.25에서 0.0056, LEACE의 α=0.75에서 0.0039). 테스트 표본이 34건뿐인 칸(여성 rapper)의 예측 한두 개가 실행마다 바뀌기 때문이며, 같은 데이터로 같은 모델을 네 번 학습한 보정 GAP이 0.084~0.089로 달랐음 (BME, α=0.25)
- 다시 실행한 결과로 표를 만들면 표 2의 65칸 가운데 16칸, 표 3의 40칸 가운데 4칸이 마지막 자리에서 달라짐 (예: BME의 보정 GAP 0.075 → 0.076)
- 본문의 결과를 낸 실행에는 본문에 싣지 않은 대조 실험이 함께 들어 있었고 같은 난수 생성기를 썼음. 같은 표본이 뽑히도록 `removal.py`, `augmentation.py`, `multigroup.py`는 그 실험이 쓰던 만큼의 난수를 먼저 흘려보냄 (주석 `random stream of the reported run`)
- `frozen.json`, `rare_cells.json`, `tradeoff_*.json`의 원본 모델은 다른 결과 파일의 원본 모델과 설정이 같지만 따로 학습한 것이라, 같은 모델의 수치가 파일마다 위와 같은 크기로 다름
- `tests/test_math.py`는 본문의 명제 1~3과 구현 (편향 행렬, 편집 연산, 순열검정, 최소 증강 계획, 사본, 지표)을 합성 데이터로 확인함: `python -m pytest tests -q`
