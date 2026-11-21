# 본문의 표와 수치

`results/*.json`에서 `python -m experiments.tables`로 생성함.

## 표 2. Bias in Bios

Bias in Bios에서의 BME 결과 표. 표의 상단 부분은 임베딩 편집 기법들의 결과이며, 표의 하단 부분은 학습 데이터 교정 기법들의 결과이다. 추가량/N은 원본 데이터 수 대비 증강시킨 데이터 수 비율이다. 소수 그룹과 다수 그룹은 각 클래스에서의 해당 그룹 재현율을 평균한 값이다.

| 방법 | 추가량/N | 정확도 (%) | 보정 GAP [95% 구간] | 소수 그룹 (%) | 다수 그룹 (%) |
|---|---|---|---|---|---|
| Original | -- | 81.9 | 0.162 [0.141, 0.180] | 65.2 | 76.0 |
| *임베딩 편집* | | | | | |
| Data matrix X (k=28) | -- | 67.4 | 0.150 [0.127, 0.168] | 57.2 | 66.2 |
| LEACE | -- | 81.8 | 0.130 [0.109, 0.148] | 67.5 | 75.3 |
| INLP (32회) | -- | 81.8 | 0.120 [0.096, 0.137] | 68.0 | 75.1 |
| INLP (직업 보정, 128회) | -- | 81.7 | 0.109 [0.084, 0.125] | 68.6 | 74.7 |
| BME (Ours, soft λ=0.25) | -- | 81.7 | 0.129 [0.108, 0.145] | 67.2 | 75.3 |
| BME (Ours, 직교) | -- | 81.2 | 0.080 [0.048, 0.098] | 69.6 | 73.5 |
| BME (Ours) | -- | 81.2 | 0.075 [0.045, 0.091] | 69.9 | 73.5 |
| BME (Ours, 모델 고정) | -- | 80.1 | 0.062 [0.028, 0.080] | 67.7 | 69.6 |
| *학습 데이터 교정* | | | | | |
| Reweighing | +0.000 | 81.4 | 0.069 [0.037, 0.086] | 70.9 | 73.1 |
| CDA (all) | +1.000 | 81.8 | 0.113 [0.089, 0.129] | 68.5 | 75.0 |
| CDA (partial) | +0.224 | 81.5 | 0.103 [0.080, 0.118] | 70.1 | 75.6 |
| BMA (Ours) | +0.224 | 81.2 | 0.075 [0.045, 0.091] | 72.0 | 74.5 |

## 표 3. 위해위험 소비자상담 내역

위해위험 소비자상담 내역 데이터 적용 결과. 표의 상단은 이진 그룹의 경우이며, 표의 하단은 다중 그룹인 경우의 결과이다.

| 방법 | 정확도 (%) | 보정 GAP [95% 구간] | 소수 그룹 (%) | 다수 그룹 (%) |
|---|---|---|---|---|
| *판매방식 2개 그룹 (일반, 비대면), 품목 15종* | | | | |
| 원본 | 86.3 | 0.205 [0.168, 0.233] | 65.7 | 79.0 |
| BME (재학습) | 83.9 | 0.146 [0.103, 0.172] | 66.4 | 75.8 |
| BME (모델 고정) | 81.8 | 0.110 [0.070, 0.132] | 65.8 | 72.4 |
| reweighing | 85.9 | 0.179 [0.135, 0.208] | 66.0 | 78.7 |
| BMA | 85.8 | 0.180 [0.137, 0.207] | 66.9 | 78.6 |
| *판매방식 3개 그룹 (일반, 인터넷, 홈쇼핑), 품목 6종* | | | | |
| 원본 | 91.2 | 0.144 [0.103, 0.169] | 82.9 | 91.7 |
| BME (재학습) | 90.0 | 0.111 [0.073, 0.136] | 84.1 | 89.9 |
| BME (모델 고정) | 88.9 | 0.085 [0.027, 0.110] | 83.3 | 88.4 |
| reweighing | 91.1 | 0.118 [0.076, 0.144] | 82.9 | 91.8 |
| BMA | 90.9 | 0.110 [0.069, 0.136] | 82.8 | 91.4 |

## 본문 수치

| 항목 | 값 | 키 |
|---|---|---|
| Bias in Bios, BME (재학습): 정확도 손실 | 0.7%p | `bios.bme.acc_drop` |
| Bias in Bios, BME (재학습): 보정 GAP 감소율 | 54% | `bios.bme.gap_reduction` |
| Bias in Bios, BME (사교 편집, 재학습): 보정 GAP | 0.075 | `bios.bme.gap` |
| Bias in Bios, BME (직교 편집, 재학습): 보정 GAP | 0.080 | `bios.bme_orth.gap` |
| Bias in Bios, BME (직교 편집, 모델 고정): 정확도 | 65.5% | `bios.bme_orth.frozen_acc` |
| Bias in Bios, BME (soft editing λ=0.25, 재학습): 보정 GAP | 0.129 | `bios.bme_soft.gap` |
| Bias in Bios, BME (모델 고정): 정확도 | 80.1% | `bios.bme_frozen.acc` |
| Bias in Bios, BME (모델 고정): 정확도 손실 | 1.8%p | `bios.bme_frozen.acc_drop` |
| Bias in Bios, BME (모델 고정): 보정 GAP 감소율 | 62% | `bios.bme_frozen.gap_reduction` |
| Bias in Bios, 데이터 행렬 X의 상위 28개 제거: 정확도 손실 | 14.5%p | `bios.x_top28.acc_drop` |
| Bias in Bios, 데이터 행렬 X의 상위 28개 제거: 보정 GAP | 0.150 | `bios.x_top28.gap` |
| Bias in Bios, 순차 순열검정이 선택한 방향의 수 (28개 중) | 22개 | `bios.k_selected` |
| Bias in Bios, 임베딩 분산 가운데 클래스 안의 그룹 차이가 차지하는 비율 | 1.09% | `bios.bias_share` |
| Bias in Bios, BMA: 추가한 표본의 양 (원본 대비) | 0.224N | `bios.bma.added` |
| Bias in Bios, BMA: 정확도 손실 | 0.6%p | `bios.bma.acc_drop` |
| Bias in Bios, BMA: 보정 GAP 감소율 | 54% | `bios.bma.gap_reduction` |
| Bias in Bios, CDA (all): 보정 GAP 감소율 | 30% | `bios.cda.gap_reduction` |
| Bias in Bios, 편향 강도 α=0, 원본 모델: 보정 GAP | 0.090 | `dose.bios.none.a0.gap` |
| Bias in Bios, 편향 강도 α=1, 원본 모델: 보정 GAP | 0.179 | `dose.bios.none.a1.gap` |
| Bias in Bios, 편향 강도 α=1, LEACE: 보정 GAP | 0.150 | `dose.bios.leace.a1.gap` |
| Bias in Bios, 편향 강도 α=1, INLP: 보정 GAP | 0.140 | `dose.bios.inlp.a1.gap` |
| Bias in Bios, 편향 강도 α=1, BME: 보정 GAP | 0.086 | `dose.bios.bme.a1.gap` |
| Bias in Bios, 편향 강도 α=1, BMA: 보정 GAP | 0.107 | `dose.bios.bma.a1.gap` |
| Bias in Bios, BMA: 데이터의 의존 때문에 늘어난 보정 GAP 가운데 줄인 비율 | 80% | `dose.bios.bma.removed` |
| 위해위험 상담, 편향 강도 α=0, 원본 모델: 보정 GAP | 0.173 | `dose.hazard.none.a0.gap` |
| 위해위험 상담, 편향 강도 α=1, 원본 모델: 보정 GAP | 0.206 | `dose.hazard.none.a1.gap` |
| 위해위험 상담, 편향 강도 α=1, LEACE: 보정 GAP | 0.200 | `dose.hazard.leace.a1.gap` |
| 위해위험 상담, 편향 강도 α=1, INLP: 보정 GAP | 0.207 | `dose.hazard.inlp.a1.gap` |
| 위해위험 상담, 편향 강도 α=1, BME: 보정 GAP | 0.135 | `dose.hazard.bme.a1.gap` |
| 위해위험 상담, 편향 강도 α=1, BMA: 보정 GAP | 0.177 | `dose.hazard.bma.a1.gap` |
| 위해위험 상담, BMA: 데이터의 의존 때문에 늘어난 보정 GAP 가운데 줄인 비율 | 88% | `dose.hazard.bma.removed` |
| Bias in Bios, 편향 강도 α=0, 원본 모델: 소수 그룹 재현율 | 65.8% | `dose.bios.none.a0.tpr_minority` |
| Bias in Bios, 편향 강도 α=1, 원본 모델: 소수 그룹 재현율 | 59.4% | `dose.bios.none.a1.tpr_minority` |
| 위해위험 상담: 실험에 쓴 상담 건수 (학습 + 테스트) | 104,552건 | `hazard.n` |
| 위해위험 상담: 품목 대분류의 수 | 15종 | `hazard.n_classes` |
| 위해위험 상담: 비대면 판매의 비율 (전체) | 18.8% | `hazard.remote_share` |
| 위해위험 상담: 품목별 비대면 판매 비율의 최솟값 | 5% | `hazard.remote_share_min` |
| 위해위험 상담: 품목별 비대면 판매 비율의 최댓값 | 40% | `hazard.remote_share_max` |
| 위해위험 상담: 임베딩 분산 가운데 클래스 안의 그룹 차이가 차지하는 비율 | 1.15% | `hazard.bias_share` |
| 위해위험 상담, 원본 모델: 보정 GAP | 0.205 | `hazard.none.gap` |
| 위해위험 상담, 원본 모델: 비대면 판매 상담의 재현율 | 65.7% | `hazard.none.tpr_minority` |
| 위해위험 상담, 원본 모델: 일반 판매 상담의 재현율 | 79.0% | `hazard.none.tpr_majority` |
| 위해위험 상담, BME (재학습): 보정 GAP 감소율 | 29% | `hazard.bme.gap_reduction` |
| 위해위험 상담, BME (모델 고정): 보정 GAP 감소율 | 46% | `hazard.bme_frozen.gap_reduction` |
| 위해위험 상담, BMA: 정확도 손실 | 0.5%p | `hazard.bma.acc_drop` |
| 위해위험 상담, BMA: 보정 GAP 감소율 | 12% | `hazard.bma.gap_reduction` |
| 위해위험 상담, BMA: 비대면 판매 상담의 재현율 변화 | +1.3%p | `hazard.bma.tpr_minority_gain` |
| 위해위험 상담, 비대면 판매가 드문 품목(학습 데이터에서 10% 미만), 원본 모델: 비대면 판매 상담의 재현율 | 66.3% | `hazard.rare.none` |
| 위해위험 상담, 비대면 판매가 드문 품목, BME (재학습): 비대면 판매 상담의 재현율 | 72.5% | `hazard.rare.bme` |
| 위해위험 상담, 비대면 판매가 드문 품목, BMA: 비대면 판매 상담의 재현율 | 72.2% | `hazard.rare.bma` |
| 위해위험 상담 (3개 그룹), 원본 모델: 보정 GAP | 0.144 | `multigroup.none.gap` |
| 위해위험 상담 (3개 그룹), BME (재학습): 보정 GAP | 0.111 | `multigroup.bme.gap` |
| 위해위험 상담 (3개 그룹), BMA: 보정 GAP | 0.110 | `multigroup.bma.gap` |
