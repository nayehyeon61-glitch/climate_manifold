# Dynamics의 미래 장·hybrid·vector-only 감독 비교

브랜치: `feature/dynamics-vector-only`

기존 예측 경로와 Dynamics Encoder `E_d`를 유지하면서 **Dynamics 보조 경로의 감독
방식**을 선택합니다. 주실험은 `field`와 `field_latent`이며, 두 실험은 동일한
자기회귀 latent 전이 `F_d`와 미래 장 Decoder `D_d`를 사용합니다. 차이는 latent
일치 손실의 유무뿐입니다. 기존의 lead-conditioned 직접 보조 Decoder는 기본값
`direct`로 보존합니다. 기존 `direct`와 새 `latent_rollout`을 비교하면 구조도 달라지므로,
그 차이를 latent 일치 손실만의 효과로 해석하면 안 됩니다.

새 `vector_only`는 **Dynamics 보조 경로에서 미래 latent만 감독**하는 선택입니다.
기존 `field`/`field_latent` 비교는 유지하며, runner에서는 `INCLUDE_VECTOR_ONLY=1`로
새 조건을 추가합니다. 전체 모델의 미래 장 학습을 제거하지는 않습니다. 주 예측 모델
`M`의 미래 장 손실과 `E_s`의 관측 복원·통계 손실은 그대로 공동 학습합니다.

## 구조와 학습 신호

관측 시퀀스를 `E_d`로 encode한 마지막 latent에서 시작하여, archive의 시간 간격마다
같은 `F_d`를 반복 적용합니다. 미래 장 정답이나 미래 latent를 rollout 입력에 넣지 않습니다.

\[
z_t=E_d(X_{\le t})_{\mathrm{last}},\qquad
\widehat z_{t+k}=F_d^{(k)}(z_t),\qquad
\widetilde x_{t+k}=D_d(\widehat z_{t+k}).
\]

`F_d`는 이번에 추가한 **보조 경로 전용** 전이 모듈입니다. ClimODE·NeuralODE 등
기존 예측 모델 `M`을 대체하지 않습니다. 실제 예측은 이전과 동일합니다.

| 경로 | 실제 예측 연결 | Dynamics 보조 경로 |
|---|---|---|
| Latent | `(E_s + P(E_d)) → M → D_forecast` | `E_d → F_d 반복 → D_d` |
| Guided 1단 | `T2(Raw, E_s, E_d) → M` | 동일 |
| Guided 2단, 선택 | `T2(Raw, T1(E_s, E_d)) → M` | 동일 |

`field`는 `D_d`의 미래 장에 기존 면적 가중 L²와 선택 통계 손실을 적용합니다.
`field_latent`는 동일한 미래 장 손실에 다음 일치 손실을 추가합니다.
`vector_only`는 Dynamics 보조 경로에서 이 latent 손실만 사용합니다.

\[
z^*_{t+k}=\operatorname{stopgrad}
\left[E_d(X^{\mathrm{truth}}_{\le t+k})_{\mathrm{last}}\right],\qquad
\mathcal L_{\mathrm{latent}}=
\frac1{KD}\sum_{k=1}^K\|\widehat z_{t+k}-z^*_{t+k}\|_2^2.
\]

여기서 `D`는 Dynamics 공간 latent tensor의 전체 좌표 수이며, batch에서도 평균합니다.

미래 정답 표현도 **동일한 `E_d`**로 계산합니다. 각 미래 시점을 끝으로 하는 정답 창은
관측 입력과 같은 길이와 history stride를 사용하며, 창마다 Encoder의 상태를 새로
시작합니다. 예를 들어 일별 자료의 관측 창이 `t-5,…,t`라면 2일 뒤 정답 창은
`t-3,…,t+2`입니다. 6시간 archive에서 하루 간격 history를 사용하면 미래 창도
같은 하루 간격을 유지합니다. 정답이므로 포함되는 미래 관측은 **손실/진단 전용**이며
실제 예측 경로와 `F_d`의 입력에는 전달하지 않습니다.

정답 쪽에는 gradient를 끊습니다. 그러나 `E_d`는 실제 예측 경로와 선택한 Dynamics
보조 손실로 계속 학습하므로 정답 좌표도 업데이트마다 달라지는 online target입니다. EMA teacher나
고정 Encoder를 사용하지 않으며, stop-gradient만으로 collapse 방지를 보장하지 않습니다.
`vector_only`에서는 Dynamics 미래 장 감독이 없어지므로 좌표 variance와 RMS norm을
특히 함께 확인해야 합니다. 주 예측 경로가 Raw나 `E_s`에 의존해도 성능을 낼 수 있어,
주 예측 손실이 남아 있다는 사실만으로 `E_d`의 collapse가 방지된다고 볼 수 없습니다.
여기서 vector는 `E_d`의 공간 latent feature tensor를 뜻하며 바람 벡터나
물리적 시간 미분을 정답으로 두는 방식은 아닙니다. 기존 공간 통계 보조 손실은
별도 `E_s`를 감독합니다. `E_d`의 표현 의미를 직접 붙잡는 것은 이번 Dynamics
미래 장·통계 손실(`field`/`field_latent`)과 실제 예측 경로에서 전달되는 gradient입니다.

\[
\mathcal L_{\mathrm{total}}=
\mathcal L_{\mathrm{forecast}}+\mathcal L_{\mathrm{observed}}
+\lambda_f\mathcal L_{d,\mathrm{field}}
+\lambda_s\mathcal L_{d,\mathrm{stat}}
+\lambda_\Delta\mathcal L_{d,\mathrm{tendency}}
+\mathbf1_{\{\mathrm{field\_latent},\mathrm{vector\_only}\}}\lambda_z\mathcal L_{\mathrm{latent}}.
\]

기본은 `λf=0.1`, `λs=0.1`, `λΔ=0`, hybrid/vector-only의 `λz=0.1`입니다.
`field`/`field_latent`에서는 미래 장·통계 가중치를 모두 양수로 유지합니다.
`vector_only`에서는 `λf=λs=λΔ=0`이며 보조 `D_d`를 호출하지 않고 동결합니다.
이 모드의 `D_d`는 학습된 미래 장 Decoder가 아니므로 물리장 예측기로 평가하지 않습니다.
`E_d/F_d`의 구조는 다른 rollout 조건과 같지만, 보조 Decoder의 학습 여부와 계산량도
함께 달라집니다. 따라서 vector-only 비교를 latent 손실 하나만 추가한 효과로 해석하지 않습니다.
공간 Encoder·통계 보조 경로의 손실은
기존대로입니다. 하나의 optimizer로 공동 학습하며, `F_d`의 다단계 출력에 역전파합니다.
추론의 일반 예측에서는 학습용 `F_d/D_d`를 호출하지 않습니다. 별도 Dynamics 평가를
요청했을 때만 보조 rollout을 실행합니다. 추가 latent 손실이 자동으로 추론을 빠르게
만들거나 예측 성능을 보장하지는 않습니다.

이 설계는 [LE-PDE §3.2](https://cs.stanford.edu/~jure/pubs/learning-pde-neurips22.pdf)의
미래 장과 다단계 latent 일관성을 함께 감독하는 아이디어를 참고한 비교이며, 해당
논문의 구조·정규화·학습 조건을 그대로 재현하는 구현은 아닙니다.

## 선택 옵션

| 목적 | train 옵션 |
|---|---|
| 기존 직접 미래 장 보조 Decoder 유지 | `--dynamics-auxiliary-architecture direct --dynamics-supervision field` |
| 새 latent 전이, 미래 장 감독만 | `--dynamics-auxiliary-architecture latent_rollout --dynamics-supervision field` |
| 동일한 전이 + 미래 latent 감독 | `--dynamics-auxiliary-architecture latent_rollout --dynamics-supervision field_latent --dynamics-latent-weight 0.1` |
| Dynamics 미래 latent만 감독 | `--dynamics-auxiliary-architecture latent_rollout --dynamics-supervision vector_only --dynamics-latent-weight 0.1` |

`direct`가 기존 기본값입니다. `field`/`direct`에 latent 가중치를 명시하거나
`direct + field_latent/vector_only`를 선택하면 오류로 처리합니다. `vector_only`를
CLI로 선택하면 Dynamics의 reconstruction/statistical/tendency 가중치 기본은 모두
0이 됩니다. 이 세 가중치에 양수를 명시하면 오류로 처리합니다. 다른 두 감독 모드의
기본 가중치는 유지됩니다. 비교 runner는 vector-only arm에서만 이 세 가중치를
명시적으로 0으로 덮어씁니다. **전체 forecast/reconstruction/statistical 가중치를 0으로
바꾸면 다른 실험이 됩니다.** 통계 손실은 여전히
`--dynamics-statistical-loss w2|kl_entropy|signed_measure`로 선택합니다.
공간 통계 손실은 별도 `--statistical-loss`입니다. W2는 값의 주변분포,
KL-entropy는 histogram 분포 비교이고, signed measure는 정규화 기준의 양/음
공간 모양과 총량을 비교합니다. signed의 기준은 물리적 0 Pa가 아닙니다.
`vector_only`에서는 Dynamics 통계 손실을 계산하지 않으며, 위 선택은 `E_s`의
관측 통계 감독과 다른 arm의 미래 장 통계 감독에서 유효합니다.

## 비교 실행과 평가

새 `scripts/run_dynamics_consistency_comparison.sh`는 기본적으로 다음 네 arm만
학습합니다. **4개 arm × 8개 예측 모델 × 3개 seed = 96회**, batch 16입니다.

| arm | 예측 경로 | Dynamics 감독 |
|---|---|---|
| `latent_dynamics_field` | Latent | 미래 장 |
| `latent_dynamics_field_latent` | Latent | 미래 장 + latent |
| `guided_dynamics_field` | Guided 1단 | 미래 장 |
| `guided_dynamics_field_latent` | Guided 1단 | 미래 장 + latent |

`INCLUDE_TWO_STAGE=1`이면 `guided_dynamics_two_stage_field`와
`guided_dynamics_two_stage_field_latent`도 추가되어 **144회**가 됩니다.
두 단계 비교에서도 같은 두 손실 조건을 유지합니다.

`INCLUDE_VECTOR_ONLY=1`이면 `latent_dynamics_vector_only`와
`guided_dynamics_vector_only`를 추가합니다. 따라서 기본 2개 경로에서는
**6개 arm × 8개 M × 3개 seed = 144회**이며, `INCLUDE_TWO_STAGE=1`이면
`guided_dynamics_two_stage_vector_only`까지 총 **9개 arm × 8개 M × 3개 seed = 216회**입니다.
이 옵션의 기본값은 `0`이며 `INCLUDE_DYNAMICS_CONSISTENCY=1`이 필요합니다.

새 wrapper는 `INCLUDE_DYNAMICS=1`, `INCLUDE_DYNAMICS_CONSISTENCY=1`,
`CONSISTENCY_ONLY=1`, `INCLUDE_TWO_STAGE_BYPASS=0`을 기본으로 사용합니다.
`CONSISTENCY_ONLY=0`이면 기존 raw·latent·guided·direct dynamics 비교군과 함께
실행할 수 있습니다. 기존 runner의 기본 행렬은 바뀌지 않습니다. Consistency-only
모드에서는 기존 direct checkpoint의 T1 우회 평가를 만들지 않으므로
`INCLUDE_TWO_STAGE_BYPASS=1`은 거절합니다.

주평가는 **기존 예측 모델 `M`의 미래 물리장**에 대해 같은 시점·변수·lead에서
RMSE·ACC를 비교합니다. 새 arm에는 `--dynamics-diagnostics`를 자동 적용하여
별도로 latent 오차와
norm·variance를 기록합니다. 서로 학습한 Encoder의 좌표·스케일이 다를 수 있으므로,
latent 오차만으로 다른 실험의 우열을 판정하지 않습니다.
`field`/`field_latent`에서는 보조 `F_d/D_d` rollout의 물리장 오차와 signed 모양/총량도
계산합니다. `vector_only`는 동결된 `D_d`의 물리장 점수를 제공하지 않습니다.
세 방식의 공통 비교 대상은 **동일한 일반 예측 모델 `M`의 미래 물리장**입니다.
학습 로그의 latent variance는 한 batch의 표본과 lead를 함께 묶어 계산합니다.
평가 진단은 각 lead를 고정한 뒤 여러 origin에 걸친 variance를 계산하므로 두 값을
동일한 지표처럼 직접 비교하지 않습니다. RMS norm과 variance 감소는 collapse나
단순 스케일 축소의 징후를 살펴보기 위한 진단이며, 그 자체로 collapse 판정은 아닙니다.

기본 진단은 학습 horizon과 같습니다. `AUX_ROLLOUT_STEPS=10`처럼 늘리면 같은
`F_d`를 학습 범위 밖까지 반복하여 안정성을 진단합니다. 이때 **일반 예측 모델 `M`의
lead는 그대로**이며, 더 긴 정답이 존재하는 공통 origin으로 평가를 제한합니다.
같은 origin을 유지하기 위해 확장 진단은 `CONSISTENCY_ONLY=1`에서만 허용합니다.
`AUX_ROLLOUT_STEPS`는 학습 horizon 이상이어야 합니다. 연속시간 ODE나 확률적
SDE rollout을 새로 구현한 것은 아닙니다.

## 브랜치 받기 → 전체 학습 → test → 결과 그래프

할당된 GPU 세션에서 아래 전체 명령을 실행합니다. 코드·환경·캐시·전처리·결과는
`/lustre/home/yehyeon` 아래에 쓰고 원본 ERA5는 지정 위치에서 읽기만 합니다.
기본 격자는 16×32, 연속 6일 관측 → 미래 1~5일입니다.

```bash
bash <<'BASH'
set -euo pipefail
cd /lustre/home/yehyeon
export DAILY_WORK
DAILY_WORK=$(mktemp -d "$PWD/climate_manifold_vector_XXXXXXXX")
export ERA5_ROOT=/lustre/home/mahmed/ERA5_0p25_DAILY
mkdir -p "$DAILY_WORK"/{tmp,cache,logs}
export TMPDIR="$DAILY_WORK/tmp" TMP="$DAILY_WORK/tmp" TEMP="$DAILY_WORK/tmp"
export XDG_CACHE_HOME="$DAILY_WORK/cache" PIP_CACHE_DIR="$DAILY_WORK/cache/pip"
export TORCH_HOME="$DAILY_WORK/cache/torch" HF_HOME="$DAILY_WORK/cache/huggingface"
export CUDA_CACHE_PATH="$DAILY_WORK/cache/cuda" TRITON_CACHE_DIR="$DAILY_WORK/cache/triton"
export PYTHONPYCACHEPREFIX="$DAILY_WORK/cache/pycache" MPLCONFIGDIR="$DAILY_WORK/cache/matplotlib"
export NUMBA_CACHE_DIR="$DAILY_WORK/cache/numba" PYTHONNOUSERSITE=1

git clone --single-branch --branch feature/dynamics-vector-only \
  https://github.com/nayehyeon61-glitch/climate_manifold.git "$DAILY_WORK/code"
cd "$DAILY_WORK/code"
python3 -m venv "$DAILY_WORK/venv"
source "$DAILY_WORK/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'xarray==2024.11.0' -e '.[forecast,era5,test,plots]'

export PYTHON="$DAILY_WORK/venv/bin/python" DEVICE=cuda
export MODELS="mlp neural_ode climode convlstm simvp fourcastnet climax transformer"
export SEEDS="7 19 43" EPOCHS=20 BATCH_SIZE=16
export INCLUDE_DYNAMICS=1 INCLUDE_DYNAMICS_CONSISTENCY=1 CONSISTENCY_ONLY=1
export INCLUDE_VECTOR_ONLY=1
export INCLUDE_ZERO_DYNAMICS=0 INCLUDE_TWO_STAGE=0 INCLUDE_TWO_STAGE_BYPASS=0
export DYNAMICS_ENCODER=conv_gru DYNAMICS_HIDDEN_DIM=32 DYNAMICS_CHANNELS=8
export DYNAMICS_LATENT_WEIGHT=0.1
export STATISTICAL_LOSS=signed_measure DYNAMICS_STATISTICAL_LOSS=signed_measure
export DYNAMICS_RECONSTRUCTION_WEIGHT=0.1 DYNAMICS_STATISTICAL_WEIGHT=0.1 DYNAMICS_TENDENCY_WEIGHT=0
export GUIDE_INTERACTION_DEPTH=1 GUIDE_FUSION_DEPTH=2
export INCLUDE_ZERO_GUIDE=1 INCLUDE_GUIDED_FORECAST_ONLY=0
export CONSTRAINT_DECODER=separate_surface_and_information
export VARIABLE_CONDITIONING=0 GUIDE_DIRECT_INFORMATION=1
export STATISTICAL_FLOW_WEIGHT=0 CONDITIONAL_FLOW_WEIGHT=0
export START_DATE=1979-01-01 END_DATE=2025-12-31
export TARGET_LAT_POINTS=16 TARGET_LON_POINTS=32
export HISTORY_STEPS=6 HORIZON_STEPS=5 MAX_WINDOWS=0 MAX_CASES=0 ORIGIN_STRIDE=1
export RUN_PREFLIGHT_TESTS=1 EVALUATE_TEST=1 MAKE_PLOTS=1 OMP_NUM_THREADS=4
export GPU_MAX_UTILIZATION=10 GPU_MAX_MEMORY_PERCENT=10 GPU_MIN_FREE_GIB=8 GPU_WAIT_SECONDS=0
unset ARCHIVE INFO A_CHECKPOINT RUN RESUME AUX_ROLLOUT_STEPS
printf 'Work directory: %s\n' "$DAILY_WORK"
bash scripts/run_dynamics_consistency_comparison.sh
BASH
```

이중 Transformer도 비교하려면 위의 `INCLUDE_TWO_STAGE=0`을 `1`로 바꾸세요.
위 명령은 vector-only를 포함한 144회 학습입니다. `INCLUDE_VECTOR_ONLY=0`이면
기존 field/hybrid의 96회 비교로 돌아갑니다.
장기 진단은 마지막 `unset`에서 `AUX_ROLLOUT_STEPS`를 제외하고, 예를 들어
`export AUX_ROLLOUT_STEPS=10`을 추가하면 됩니다.

GPU guard는 할당·가시 GPU 중 사용률/메모리 점유율 각각 10% 이하, 여유 8 GiB 이상이고
다른 compute process가 없는 장치를 확인합니다. 학습 중 사용량을 10%로 제한하는
기능은 아닙니다. 조건에 맞는 장치가 없으면 시작하지 않습니다. 학습은 순차 실행합니다.
설정 선택 중에는 `EVALUATE_TEST=0`으로 test set을 남겨두세요. 비교하는 8개 `M`은
데이터/격자에 맞춘 adaptation이며 공식 pretrained checkpoint 성능 재현은 아닙니다.

기존 prepared archive를 재사용하려면 새 RUN을 정하고 다음처럼 실행할 수 있습니다.

```bash
export ARCHIVE="$DAILY_WORK/prepared/surface.npz"
export INFO="$DAILY_WORK/prepared/information.npz"
export RUN="$DAILY_WORK/runs/consistency-$(date +%Y%m%d-%H%M%S)"
bash scripts/run_dynamics_consistency_comparison.sh
```

## 결과와 재개

- `comparison.json`, `comparison.test.json`: validation/test 비교.
- `comparison*.dynamics-consistency-effects.csv`: 같은 예측 모델·경로·구조에서
  field 대비 field_latent의 효과. 기존 direct dynamics와 합쳐 집계하지 않습니다.
- `comparison*.dynamics-vector-effects.csv`: vector-only와 field/hybrid 사이의
  **주 예측 모델 `M`의 물리장 점수** 비교. `dynamics_vector_gain_percent` 그래프와
  함께 보며, 서로 다른 감독·Decoder 학습 조건의 비교로 해석합니다.
- `plots/validation/`, `plots/test/`: 비교 PNG/PDF와 manifest.
- 각 평가 JSON의 `dynamics_evaluation_contract`: 실제 보조 rollout 기간과 시간 간격.
  `dynamics_auxiliary_config`에는 감독 방식·가중치·target semantics가 저장됩니다.
- 각 체크포인트의 미래 물리장 예측은 원래 형식으로 저장하며, 보조 진단은 구분합니다.

`RESUME=1 RUN="$DAILY_WORK/runs/<기존 실행 폴더>"`는 같은 설정·소스·prepared
자료의 완료 checkpoint와 평가를 재사용합니다. 가중치, arm, 진단 horizon을 바꾸면
새 RUN이 필요합니다. 진단 계약과 checkpoint hash도 검증합니다. 중단된 optimizer
상태를 이어 학습하는 기능은 아닙니다. 서로 다른 학습 seed는 예측 ensemble이 아닙니다.

연결 테스트와 합성 자료의 짧은 실행은 구현의 동작을 검증합니다. 실제 ERA5 전체 학습,
GPU 처리량 또는 latent 일관성이 예측력을 개선한다는 연구 결과를 대신하지 않습니다.
