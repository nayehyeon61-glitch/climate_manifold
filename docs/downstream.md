# Climate Manifold와 예측기의 공동 학습 실험

기본 주실험은 **encoder → latent 예측기 → decoder 전체의 공동 학습**입니다.
미래 기상장 손실이 decoder·예측기·encoder까지 전달됩니다. A checkpoint나 Plain AE의
사전학습은 필요하지 않습니다. Hydra B/C, MoE, filtering은 포함하지 않습니다.
세부 손실과 단일 실행 예제는 [joint_training.md](joint_training.md)를 참고하세요.

## 주실험: 직접 예측과 공동 학습의 3-way 비교

| 비교군 | 예측 경로 | 학습 목표 |
|---|---|---|
| Raw | **`data → spatial model → future fields`**, E/D 없음 | 미래 state + tendency |
| Forecast only | `data → E → 공간 Neural ODE / latent ClimODE → D` | 미래 state + tendency + 현재 reconstruction |
| **Climate Manifold** | **Forecast only와 동일한 E–F–D 구조와 입력** | 같은 공통 목표 + 물리·정보 제약 |

기본 Neural ODE와 ClimODE 각각 위 세 비교군을 실행합니다. Raw는 같은 공간 예측기 core를
원본 기상 격자에 직접 적용하며, 모델에 별도 manifold encoder/decoder가 없습니다.
모델 자체의 CNN feature/context 처리는 유지합니다. MLP는 선택적으로 실행할 수 있습니다.
**Seed마다 모델 전체를 새로 학습**하고, 같은 seed의 E–F–D off/on pair는
같은 초기화와 학습 예산을 사용합니다.
Reconstruction-only AE와 미래 감독을 받은 A를 비교하는 기존 실험보다 추가 제약의 효과를
직접 검증합니다. `forecast_only`라는 이름에도 reconstruction 보조 손실은 포함됩니다.
Raw에는 복원 경로가 없으므로 reconstruction 및 정보/물리/PINN 보조 손실을 계산하지 않습니다.
Raw와 latent 모두 동일한 관측 history, 시간 feature, origin의 상층/지형 정보에 접근합니다.
Raw에서는 origin 정보를 예측기 context에 직접 넣습니다. 미래 정보는 어느 모델에도 입력하지 않습니다.

| 비교 | 해석할 수 있는 효과 |
|---|---|
| Climate Manifold vs Raw | 표현·용량·추가 감독을 합친 전체 모델의 예측 개선 |
| Forecast only vs Raw | E/D 표현·공간 축소·용량 및 reconstruction 감독의 합친 효과 |
| Climate Manifold vs Forecast only | 같은 E/F/D에서 추가 물리·정보 제약의 효과 |

Raw와 latent의 채널·해상도·parameter 수가 달라 완전한 capacity matching은 아닙니다.
보고서의 parameter 수와 실행 비용을 함께 제시하고, Raw 대비 차이를 PINN만의 효과로 해석하지 않습니다.

`ManifoldBridge`가 정규화·인코딩·디코딩 계약을 연결합니다. 새 fresh 주실험은
**공간 CNN E/D와 `[C_z, ceil(H/f), ceil(W/f)]` 잠재장**을 사용합니다. 기본 채널 32,
축소 배율 2이므로 18×36 입력에서는 32×9×18 = 5,184개 좌표입니다. 동일한 전역
64차원 벡터를 reshape한 것이 아닙니다. 내부 전달/저장용 flat 배열도 알려진 공간 shape로
복원하여 연산합니다. 전체 관측 latent history를 조건으로 미래 latent를 전개하고,
공통 decoder가 각 lead를 물리 기상장으로 복원합니다.

공간 Neural ODE는 CNN vector field를 적분합니다. Latent ClimODE는 학습한 transport와
velocity 동역학을 잠재 격자에 적용하는 adaptation입니다. E/D와 입력은 두 계열에서
일치하지만 예측기 자체는 다르므로 동일 parameter 수를 주장하지 않습니다. 각 계열의
`none/full` 대조군은 같은 구조·초기화·예산을 사용합니다.

주실험에서는 `--anchor none`을 강제하여 원본 격자의 복원 잔차로 bottleneck을 우회하지
않습니다. 출력이 decoder의 상에 있다는 사실만으로 엄밀한 smooth manifold나 물리적
타당성이 보장되지는 않습니다. 미래 데이터는 loss/사후 진단의 목표이며 forward 입력이 아닙니다.

## 데이터·학습 조건

- `--initialization fresh`가 기본입니다. A 없이 archive에서 구조/정규화/분할 계약을 만듭니다.
  `--a-checkpoint`를 제공한 fresh 실행도 데이터/정규화/분할 계약을 재사용하면서
  공간 E/D를 새로 만듭니다. A 가중치 재사용은 `--initialization pretrained`로 명시하며
  joint에서는 이후 함께 업데이트합니다. 전역 A 가중치의 spatial 변환은 지원하지 않습니다.
- 모든 비교군에서 같은 archive·정보 파일·정규화·시간 분할·horizon을 사용합니다.
  학습은 `train`, 선택은 **`calibration`의 공통 미래 state MSE**, 개발 평가는 `validation`,
  최종 평가는 `test`입니다. 독립 A의 `expert_validation` 선택 경로와 구분합니다.
- 과거 history를 인코딩할 때 **origin 시점 정보**를 사용합니다. Raw도 동일한 origin 정보를
  예측기 context로 사용합니다. 미래 상층 정보는 해당 loss의 label만 되고 예측 조건에 들어가지 않습니다.
- `none`과 `full`은 같은 입력/구조를 사용합니다. `none`은 추가 surface physics,
  information, static, distribution, PINN을 끄고 common reconstruction을 유지합니다.
- Raw와 latent는 입력 차원·구조가 달라 같은 hidden width라도 parameter 수가 다릅니다.
  Raw 대비 차이를 PINN만의 인과적 효과로 해석하지 않습니다.
- Joint pair의 여러 seed는 표현과 예측기 초기화 변동을 모두 포함합니다.
  Pretrained를 같은 A에서 시작하면 A 사전학습 seed 변동까지 포함하는 것은 아닙니다.
- 기본 latent 미래 예측은 결정론적입니다. Optional spatial distribution loss는
  공간 분포 진단이며 ensemble CRPS나 시나리오 보정을 학습하는 loss가 아닙니다.

비교기는 데이터/초기화 계약, origin·lead, 실험 종류 및 학습 조건을 확인합니다.
학습된 encoder 가중치가 서로 같아야 한다는 조건은 joint 비교에 적용하지 않습니다.
주실험과 보조 ClimODE 결과는 같은 paired-effect 표에 섞지 않습니다.

## 실행

```bash
python -m pip install -e '.[test,era5]'
export ARCHIVE=/absolute/path/to/surface.npz
export INFO=/absolute/path/to/information_pinn_shards
export RUN=runs/joint_comparison_001
export DEVICE=cuda EPOCHS=20 BATCH_SIZE=2 SEEDS='7 19 43'
export PINN=1
bash scripts/run_model_comparison.sh
```

PINN용 데이터가 없으면 `PINN=0`으로 실행합니다. Surface-only는 `INFO`를 unset합니다.
기본은 **2 predictor families × 3 비교군 × 3 seeds = 18회 학습**입니다.
주실험에 ClimODE constants는 필요하지 않습니다. `RUN`은 새 경로여야 합니다.

| 환경 변수 | 기본값·역할 |
|---|---|
| `MODELS / SEEDS` | `neural_ode climode` / `7 19 43` |
| `TRAINING_MODE / INITIALIZATION` | `joint` / `fresh` |
| `A_CHECKPOINT` | 선택적 데이터·구조 계약; `pretrained`일 때 가중치 재사용 |
| `EPOCHS / LEARNING_RATE` | 20 / 0.001 |
| `LATENT_LAYOUT` | fresh에서 `spatial`; pretrained는 checkpoint 표현 상속 |
| `LATENT_CHANNELS / SPATIAL_DOWNSAMPLE / SPATIAL_HIDDEN_DIM` | 공간 잠재 채널 32 / 공간 축소 배율 2 / E·D 은닉 채널 64 |
| `HIDDEN_DIM` | 예측기 은닉 폭 128 |
| `MANIFOLD_DIM / MANIFOLD_HIDDEN_DIM` | `LATENT_LAYOUT=global`일 때만 전역 latent 64 / E·D 폭 512 |
| `CLIMODE_STEP_HOURS` | latent ClimODE 적분 간격 1h; 출력 간격과 구분 |
| `LATENT_MAX_SPEED / LATENT_MAX_ACCELERATION` | latent 수송 속도 상한 2 셀/일 / 내부 속도 좌표 변화율 상한 1/일; 해상도에 따라 조정 |
| `HISTORY_STEPS / HISTORY_STRIDE` | 6 / 4; 6시간 자료에서 입력 간격 24시간 |
| `HORIZON_STEPS` | 20; 출력 간격 6시간, 총 120시간 |
| `RECONSTRUCTION_WEIGHT / TENDENCY_WEIGHT` | 공통 loss 0.1 / 0.1 |
| `PHYSICS_WEIGHT / INFORMATION_WEIGHT / STATIC_WEIGHT` | full에서 0.01 / 0.1 / 0.05 |
| `DISTRIBUTION_WEIGHT` | 0; 선택적 공간 분포 매칭 |
| `PINN / PINN_LEVELS / PINN_WEIGHT` | 0 / `500 850` / PINN config 가중치 |
| `INCLUDE_RAW` | 1; Neural ODE·ClimODE의 직접 예측 비교군 포함. 0이면 E–F–D off/on만 실행 |
| `RAW_BACKEND` | 생략 시 joint spatial은 `matched`, global/frozen은 `legacy`; 원본 vendor ClimODE는 별도 보조 runner 사용 |
| `WINDOW_STRIDE / MAX_WINDOWS` | 4 / 0(전체 창) |
| `ORIGIN_STRIDE / MAX_CASES` | 1 / 0(전체 평가 origin) |

`MODELS=neural_ode` 또는 `MODELS=climode`로 한 계열만 실행할 수 있습니다.
`ANCHOR=origin`은 주실험에서 거부합니다. 기존 전역 표현은
`LATENT_LAYOUT=global MODELS='mlp neural_ode'`로 선택합니다. 전역 latent에는 ClimODE를
연결하지 않습니다. Pretrained 실행은 checkpoint 구조를 상속하며, 전역 checkpoint에
`LATENT_LAYOUT=spatial`을 지정하면 거부합니다. Fresh + A checkpoint는 기존 데이터/history
계약을 쓰면서 공간 표현으로 새로 초기화할 수 있습니다.
Checkpoint에는 학습된 E/F/D와 통계가 포함되므로 원본 A 경로 없이 재평가할 수 있습니다.
학습 재개를 위한 optimizer/RNG 복구는 지원하지 않습니다.

## 평가와 latent 진단

**모델 간 우열은 같은 물리 단위의 기상장 점수로 비교**합니다.

주요 평가 기준은 이제 **ClimODE 방식의 변수·lead별 RMSE/ACC와 확률 출력의 CRPS**입니다.
`scores.climode`에 사례별 점수 평균·표준편차를 저장하고 Raw ClimODE 대비 개선율을
추가합니다. [정확한 집계·공식 코드와의 차이·실행 방법](climode_evaluation.md)을 참고하세요.
아래 기존 pooled 점수와 dynamics 진단도 보존합니다.

| 질문 | 지표·판독 |
|---|---|
| 기상장을 더 잘 예측하는가? | 변수별·6h lead별 면적 가중 RMSE/MAE/bias, 전체 pooled RMSE |
| Persistence보다 나은가? | `1 − RMSE_model / RMSE_persistence`; 양수면 개선 |
| anomaly 패턴을 유지하는가? | ACC; train의 격자별 고정 평균 기준, 계절 climatology와 구분 |
| 첫 lead 이후 정체하는가? | 시간당 tendency RMSE, predicted/true tendency RMS 비율 |
| 대규모 평균·바람은 맞는가? | field-mean RMSE, U10/V10으로 계산한 풍속 RMSE |
| 안정적인가? | 전체 rollout 유한 예측 비율, 실패 origin 목록 |
| 비용은 얼마인가? | trainable/total parameter, 학습·추론 시간, CUDA peak memory |

Latent 예측 실험에는 별도 `latent_diagnostics`가 추가됩니다. **먼저 예측을 완료한 후**
실제 미래 기후장을 같은 encoder와 같은 origin 정보로 인코딩하여 진단용 목표를
만듭니다. 이것은 사후 평가이며 예측 입력이나 미래 정보 condition이 아닙니다.

| 진단 필드 | 의미 |
|---|---|
| `latent_rmse`, `latent_persistence_rmse` | 예측 latent 오차와 마지막 latent 유지 기준선 |
| `latent_tendency_rmse_per_hour`, `latent_tendency_amplitude_ratio` | latent 변화량 정확도와 정체 여부 |
| `latent_cycle_rmse` | 예측 latent를 decode→encode했을 때의 좌표 차이 |
| `future_reconstruction_normalized_rmse` | 실제 미래를 encode→decode한 재구성 오차; 예측 점수 아님 |
| `projected_persistence_normalized_rmse` | origin 복원장을 유지한 기상장 오차 |
| `forecast_vs_reconstructed_target_normalized_rmse` | 예측장과 미래 재구성장 사이 오차 |

각 공동 학습 모델은 **서로 다른 좌표계**를 학습하므로 latent RMSE 값으로 두 표현의
우열을 직접 매기지 않습니다. 미래 재구성 오차도 수학적으로 증명된 예측 오차의
하한이 아닙니다. 실제 truth에 대한 예측 오차와 함께 표현 손실·동역학 손실을
살펴보는 진단입니다. 물리 잔차 평가가 이 진단에 자동으로 포함되지는 않습니다.

출력은 모델별 `.validation.json`, 첫 성공 origin의 `.forecast.npz`,
그리고 `comparison.json/.csv`입니다. Latent 실험의 NPZ에는 물리 단위 예측·truth와
`predicted_latent`, `origin_latent`, `diagnostic_target_latent`도 포함됩니다.
추론 시간에는 미래 재구성 audit를 제외하고, 전체 평가 시간은 별도로 기록합니다.
`paired_effects / paired_summary`는 같은 모델·seed에서 full vs forecast_only,
forecast_only vs Raw, full vs Raw의 RMSE 감소량을 별도 `pair_key`로 집계합니다.
양수면 해당 candidate의 오차가 작습니다. `direct_comparison`과 `comparison.raw-effects.csv`는
같은 계열 Raw 대비 변수·lead별 물리 단위 RMSE skill 및 ACC 차이를 저장합니다.
기본 포함되는 Raw 비교는 같은 구조의 손실 ablation과 구분하여 보고합니다.

시간 변화량의 첫 전이는 관측 origin에서 첫 예측으로 계산하므로 재구성 오차도
포함합니다. 기존 `scores.per_variable` RMSE는 제곱 오차를 모은 뒤 제곱근을 취하며,
새 `scores.climode` RMSE는 공식 평가 코드처럼 사례별 RMSE를 평균합니다.
현재 MLP/Neural ODE와 matched Raw/latent ClimODE는 결정론적이며 CRPS는 `null`입니다.
Latent 분포를 비선형 D로 복원한 결과에 Gaussian CRPS를 임의로 적용하지 않습니다.
이 실험은 Hydra의 ensemble 경로 보정 성능을 평가하지 않습니다.

## ClimODE의 주실험과 보조 기준선

기본 주실험은 **직접 matched ClimODE + 공간 E → latent ClimODE → D의 off/on**입니다.
`scripts/run_climode_comparison.sh` 역시 기본은 같은 세 비교군을 ClimODE 한 계열에 실행합니다.
직접 비교군은 latent adaptation과 같은 transport core를 원본 기상 격자에서 학습하며
별도의 E/D와 Gaussian head가 없습니다. `--constants`도 필요하지 않습니다.
수송 계수는 물리 U/V 자체가 아니며, 정규화된 기상장에 대한 학습 계수입니다.
동일한 대략적 이동 범위를 위해 Raw의 speed 상한은 `LATENT_MAX_SPEED × SPATIAL_DOWNSAMPLE`
(raw 셀/일)로 정합니다. Effective bound는 보고서에 남기며 격자 반올림 때문에 기하학적으로 완전히
같지는 않습니다. Raw 격자의 CFL 조건 때문에 같은 horizon의 적분 횟수는 더 클 수 있습니다.

관측 history로 잠재 상태와 transport 동역학을 구성하고 미래 latent를 rollout합니다.
잠재 transport 계수는 물리적 U/V가 아니며 원본 ClimODE의 물리 보존식을 그대로 보장하지
않습니다. 정보·정적 지형·PINN 손실은 미래 latent를 복원한 정보/기상 변수에서 계산합니다.
실제 grid constants나 원본 ClimODE attention을 사용하지 않으므로 최소 15×15 제한도 없습니다.
잠재 격자에서도 수치 안정성과 표현 용량을 실제 데이터에서 확인해야 합니다.

아래 두 경로는 `CLIMODE_BRIDGES=raw` 또는 `CLIMODE_BRIDGES='raw decoded'`로 요청하는
**보조 실험**이며 주실험과 같은 paired-effect 표에 섞지 않습니다.

| 비교군 | 경로와 학습 |
|---|---|
| Legacy Raw ClimODE | `history grid → vendor ClimODE → future fields`; Gaussian head 포함 |
| Decoded ClimODE | `history → E → D → reconstructed grid → ClimODE`; E/D/ClimODE 공동 학습 |

두 번째는 **입력 격자 표현의 공동 학습**이며 `E → latent ClimODE → D` 실험이 아닙니다.
ClimODE 출력이 decoder image에 머무르도록 제약하지 않습니다. 보조 runner는 decoded에서
forecast/tendency와 observed reconstruction을 사용하며 future latent 경로가 필요한
information/distribution/PINN 손실을 적용하지 않습니다. Enriched 입력이 있다면 raw 대비
차이에는 상층 정보 접근 차이도 포함되므로 same E–F–D loss ablation과 구분합니다.

```bash
python -m pip install -e '.[forecast]'
python scripts/prepare_climode_constants.py --archive "$ARCHIVE" \
  --fields /absolute/path/to/constants.nc --output data/climode_constants.npz
export CONSTANTS="$PWD/data/climode_constants.npz"
export RUN=runs/joint_auxiliary_climode_001
CLIMODE_BRIDGES='raw decoded' bash scripts/run_climode_comparison.sh
```

보조 runner는 `--raw-backend legacy`를 명시합니다. 보조 raw/decoded ClimODE `--constants`에는 archive와 정확히 정렬된 실제 orography와 육해 마스크가 필요합니다.
기본 attention에는 최소 15×15 격자가 필요합니다. 작은 합성 격자의 코드 검증에는
`CLIMODE_ATTENTION=0`으로 끌 수 있으나 공식 attention 구성과 다른 ablation입니다. `CLIMODE_BRIDGES=raw`로 raw 기준선만
실행할 수 있습니다. 같은 환경에서 `scripts/run_climode_benchmark.sh`는 legacy Raw ClimODE와
primary joint 비교를 이어서 실행합니다.

이 legacy raw/decoded backend는 공식 [Aalto-QuML/ClimODE](https://github.com/Aalto-QuML/ClimODE) commit
`e729d23e8799ce0e075699e76d60227d848d8d0c`의 residual CNN, attention, transport PDE,
Gaussian head를 포함합니다. [MIT 라이선스·인용·수정 내역](../THIRD_PARTY_NOTICES.md)을
보존합니다. 이것은 **custom-data adaptation**이며 논문 benchmark의 재현이 아닙니다.

| 항목 | 수정·제약 |
|---|---|
| 변수 | 원본 Z500/T850/T2m/U10/V10 대신 msl/t2m/u10/v10 |
| 정규화 | train-only 격자별 mean/std; 원본 min/max와 다름 |
| 초기 transport velocity | 마지막 2장 관측의 backward tendency로 Adam/ridge 적합; 미래 미사용 |
| 시간·적분 | 내부 0.01×physical hours, 기본 1h Euler, 6h 출력 |
| Dropout/BatchNorm | ODE RHS 결정론적 유지; 가중치·affine는 학습 |
| 학습 | rollout Gaussian NLL + tendency; checkpoint 선택은 공통 state MSE |

초기 transport velocity의 내부 적합은 detach한 관측/복원장에서 실행하여 그 최적화까지
미분하지 않습니다. 예측 기상장 경로의 gradient는 E/D로 전달합니다. Transport velocity는
변수별 보존식 보조 상태이며 U10/V10 자체가 아닙니다.
Raw/decoded Gaussian head의 CRPS/NLL, coverage, spread–skill을 평가하지만 지점·lead별 주변분포이며
coherent trajectory ensemble을 보장하지 않습니다.

## 기존 frozen 비교의 재현

A 사전학습 → freeze → 예측기 학습은 선택적 legacy control로 보존합니다.

```bash
export A_CHECKPOINT=/absolute/path/to/manifold.pt
export RUN=runs/legacy_frozen_comparison_001
TRAINING_MODE=frozen bash scripts/run_model_comparison.sh
```

이 경우 `scripts/run_frozen_model_comparison.sh`가 기존 raw/Climate Manifold/Plain AE
3-way 비교를 실행합니다. Plain AE를 reconstruction-only로 먼저 학습하며, 모든 예측기 seed에
같은 A/AE를 고정합니다. 공동 학습 주실험과 다른 질문을 검증하므로 결과를 섞지 않습니다.
직접 train을 호출할 때에는 `--training-mode frozen --initialization pretrained`를 명시합니다.

## 검증 범위

```bash
python -m pytest -q
```

Gradient 전달, frozen 호환성, 미래 label의 입력 차단, checkpoint 재로드 및 동일 조건의
비교를 소프트웨어로 검증합니다. 합성 smoke 결과는 실제 ERA5 예측 우월성의 근거가 아닙니다.
