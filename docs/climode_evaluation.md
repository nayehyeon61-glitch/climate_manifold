# ClimODE를 기준으로 한 평가

평가의 중심은 **변수별·lead별 latitude-weighted RMSE↓, ACC↑**입니다.
Gaussian 확률 출력이 있는 모델에는 CRPS↓를 함께 보고합니다.
기준 모델은 **같은 데이터로 학습한 Raw ClimODE**이며 논문의 표에 있는 수치를
우리 결과의 분모로 사용하지 않습니다. Manifold 효과를 분리하기 위한 같은 계열의
Raw/Climate Manifold/Plain AE 비교도 함께 유지합니다.

## 확인한 일차 출처와 적용 범위

- [ClimODE ICLR 2024 논문, §4 및 Appendix C.3/I](https://arxiv.org/html/2404.10024v1)
- [공식 metrics 구현](https://github.com/Aalto-QuML/ClimODE/blob/e729d23e8799ce0e075699e76d60227d848d8d0c/utils.py)
- [공식 전구 평가 루프](https://github.com/Aalto-QuML/ClimODE/blob/e729d23e8799ce0e075699e76d60227d848d8d0c/evaluation_global.py)

공식 저장소 commit `e729d23e8799ce0e075699e76d60227d848d8d0c`를 기준으로
**사례별 지표를 계산한 후 평균하는 코드의 방식**을 반영했습니다.
논문의 RMSE 식과 코드의 집계 순서는 구분해야 합니다.
기존 pooled 지표도 유지하므로 두 집계 결과를 직접 확인할 수 있습니다.

| 항목 | 새 기본 보고 `scores.climode` | 공식 코드와의 관계 |
|---|---|---|
| RMSE | 물리 단위로 복원 → 각 origin·lead의 위도 가중 공간 RMSE → origin 평균/표준편차 | 공식 evaluation loop의 사례별 집계 |
| ACC | train 격자별 시간 평균을 뺀 anomaly → 각 장의 비가중 공간 평균 제거 → 위도 가중 상관 → 사례 평균/표준편차 | 공간 중심화는 공식 코드와 같음; climatology 자료는 다름 |
| CRPS | 물리 단위 Gaussian CRPS의 위도 가중 공간 평균 | 공식 함수는 min/max 정규화 단위의 CRPS 배열을 반환; 이 프로젝트는 물리 단위/위도 가중을 명시 |
| 기준 평균 | A checkpoint에 고정된 train-only mean | 원본 global script의 test-year 평균을 복제하지 않음 |
| 사례 표준편차 | population std (`ddof=0`) | 신뢰구간이나 seed 간 변동이 아님 |
| 공통 변수 | 현재 surface `msl/t2m/u10/v10` | 원논문 `Z500/T850/T2m/U10/V10` benchmark 재현 아님 |

위도 가중치는 `w_ij = cos(lat_i) / sum_ij cos(lat_i)`입니다. 이는 균일한
위경도 격자에서 면적 가중치에 해당합니다. 기존 `scores.per_variable` 지표는
격자 간격을 포함하는 면적 가중치를 유지하므로 비균일 격자에서는 차이가 날 수 있습니다.
ACC는 두 중심화 anomaly 중 하나가 상수이면 정의되지 않습니다. 이때 0으로
대체하지 않고 `null`과 `acc_valid_cases`를 기록합니다.

`rmse`, `acc`, `crps`와 각각의 `_std`, `_valid_cases`를 변수별
`aggregate`와 `by_lead`에 저장합니다. aggregate는 해당 변수 안에서 origin·lead를
평균한 값이며 서로 단위가 다른 변수들을 합쳐 단일 물리 RMSE를 만들지 않습니다.
결정론적 A drift·MLP·Neural ODE에는 CRPS를 `null`로 둡니다.

## 비교의 판정 기준

1. Raw ClimODE와 동일한 A 데이터 계약, archive·정보 checksum, split, origin, lead,
   metric protocol/climatology를 사용해야 합니다. 하나라도 다르면 비교를 거부합니다.
2. 후단 모델은 같은 forecast seed의 기준선과 연결합니다. A 단독 drift는 하나의
   고정 모델을 각 ClimODE seed와 비교하며 이를 독립적인 A 재학습으로 해석하지 않습니다.
3. 변수별·6h lead별 `1 − RMSE_candidate/RMSE_ClimODE`와 `ACC_candidate − ACC_ClimODE`를
   계산합니다. 둘 다 양수면 해당 지표에서 개선입니다. CRPS skill은 양쪽 모두
   확률 출력을 제공할 때만 계산합니다. 분모가 0이면 skill은 `null`입니다.
4. 두 모델 모두 요청한 **전체 horizon·모든 origin**에서 유한한 예측을 생성해야
   개선율을 보고합니다. 실패가 있으면 실패 비율은 남기되 그 pair의 skill을 계산하지 않습니다.
   ACC는 모든 사례에서 정의된 경우에만 차이를 계산합니다.
5. 6–36h 구간은 원논문의 주요 short-range 평가와 겹치는 시간대입니다.
   48/72/120h는 현재 프로젝트의 확장 평가로 구분합니다. 시간대가 겹친다는 사실만으로
   원논문과 동일한 데이터·프로토콜이 되지는 않습니다.

평균 오차만으로 dynamics 개선을 판단하지 않고 기존 tendency RMSE, 변화량 진폭 비율,
finite forecast fraction을 함께 확인합니다. Latent RMSE는 각 표현 내부의 진단입니다.
A 학습 손실과 checkpoint 선택 기준, 후단 calibration checkpoint 선택은 유지합니다.
이 변경은 **외부 평가 프로토콜**이며 test 점수로 checkpoint를 선택하지 않습니다.

## 실행

새 학습부터 Raw ClimODE와 후단 6개 조합을 연결하려면:

```bash
python -m pip install -e '.[forecast]'
export A_CHECKPOINT=/absolute/path/to/manifold.pt
export ARCHIVE=/absolute/path/to/surface.npz
export INFO=/absolute/path/to/information_pinn_shards
export CONSTANTS=/absolute/path/to/climode_constants.npz
export RUN=runs/climode_benchmark_001
export DEVICE=cuda SEEDS='7 19 43' HORIZON_STEPS=20
bash scripts/run_climode_benchmark.sh
```

Surface-only A는 `INFO`를 설정하지 않습니다. Constants는 archive와 정확히 정렬된
실제 orography/land-sea mask이며 `prepare_climode_constants.py`로 준비합니다.
ClimODE가 격자를 요구하므로 이 기준선은 raw grid에서 동작합니다.
주실험의 `encoder → latent model → decoder` 경로는 그대로 유지합니다.

출력은 `reference/`의 Raw ClimODE 결과, `latent/`의 후단 비교이며
`latent/comparison.climode.csv`는 변수·lead별 점수,
`latent/comparison.climode-effects.csv`는 같은 seed의 ClimODE 대비 개선율입니다.
기존 `comparison.csv`의 normalized pooled RMSE는 보조 요약으로 유지합니다.

기존 checkpoint는 **재학습 없이 재평가**하면 새 지표를 얻습니다:

```bash
python -m climate_manifold.downstream.evaluate \
  --checkpoint "$PREDICTOR_CHECKPOINT" --archive "$ARCHIVE" --information "$INFO" \
  --split validation --output runs/new_predictor.validation.json --device cuda
```

이미 재평가한 후보와 Raw ClimODE 보고서를 비교하려면:

```bash
python -m climate_manifold.downstream.climode_benchmark \
  --reports runs/new_predictor.validation.json \
  --climode-reference-reports runs/reference/climode-raw-seed7.validation.json \
  --output runs/new_climode_comparison.json
```

후보에는 `python -m climate_manifold.dynamics_evaluate`로 생성한 **A 단독
pure-drift 보고서**도 전달할 수 있습니다. A와 후단 보고서를 함께 전달하는 것도 가능합니다.
같은 origins/leads가 필요하므로 `--steps`, `--max-cases`, `--origin-stride`를 맞춥니다.
확정된 설정의 최종 평가에서만 양쪽을 모두 `--split test`로 재평가합니다.

기존 주실험 runner를 쓸 때 `CLIMODE_REFERENCE_DIR`를 설정하면 각 seed의
`climode-raw-seed<seed>.validation.json`을 찾아 같은 비교표를 추가합니다.
기존 보조 ClimODE runner는 `CLIMODE_BRIDGES=raw`로 기준선만 실행할 수 있습니다.

Raw ClimODE와 latent Neural ODE의 차이에는 표현뿐 아니라 예측기 구조 차이도
포함됩니다. 따라서 논문에서 manifold의 효과를 주장하려면 기존 같은 예측기 계열의
Raw/Plain AE 대조군을 함께 제시해야 합니다. 아직 실제 ERA5 우열을 입증한 결과는 아닙니다.

## 소프트웨어 검증

전체 테스트 117개가 통과했습니다. 사례별 RMSE와 pooled RMSE의 차이, 위도 가중치,
물리 단위 CRPS, ACC 공간 중심화·상수 anomaly 처리, 기준선 계약·실패 subset 검사를
확인했습니다. 기존 checkpoint에서 후단 8개 조합과 A drift를 120h × 2 origins로
재평가하여 동일 조건의 ClimODE 개선율 JSON/CSV 생성까지 검증했습니다.
이는 합성 데이터의 코드 검증이며 ERA5 예측력 실험 결과는 아닙니다.
