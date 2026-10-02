# Raw → M / E → M → D 모델 비교

브랜치: `feature/raw-latent-model-comparison`

예측기 **M 자체**를 교체하는 실험입니다. Transformer는 M의 한 선택지입니다.

| M (`--model`) | Raw 경로 | Latent 경로 |
|---|---|---|
| `transformer` | Raw → Transformer → 예측장 | E → Transformer → D |
| `mlp` | Raw → NN → 예측장 | E → NN → D |
| `neural_ode` | Raw → NeuralODE → 예측장 | E → NeuralODE → D |
| `climode` | Raw → ClimODE adaptation → 예측장 | E → ClimODE adaptation → D |
| `convlstm` | Raw → ConvLSTM → 예측장 | E → ConvLSTM → D |
| `simvp` | Raw → SimVP → 예측장 | E → SimVP → D |
| `fourcastnet` | Raw → AFNO → 예측장 | E → AFNO → D |
| `climax` | Raw → ClimaX → 예측장 | E → ClimaX → D |

같은 M의 두 경로를 비교하면 manifold를 포함한 전체 시스템의 이득을 평가합니다.
같은 경로에서 M을 Transformer/NN/NeuralODE 등으로 바꾸면 예측기 선택의 영향을 평가합니다.
입력 해상도·채널·전체 파라미터 수는 경로별로 다르며 보고서에 기록합니다.
따라서 파라미터 수까지 통제한 Transformer attention만의 효과를 입증하는 실험은 아닙니다.

`mlp`의 matched spatial 구현은 공간 convolution 기반 rate network와 Euler 갱신입니다.
ClimODE는 latent/raw 공간에 맞춘 transport adaptation이며 공식 원본 benchmark가 아닙니다.
FourCastNet/ClimaX도 공식 backbone을 현재 변수·격자에 맞춘 처음부터 학습하는 구현입니다.
ClimaX는 마지막 관측장, 다른 모델은 각 구현에 따른 관측 이력을 사용합니다.
GraphCast는 현재 latent 연결이 없어 이 두 경로 matrix에서 제외됩니다.

## 손실과 학습

- 예측: `E → M → D_forecast` 전체를 공동 학습합니다. E를 먼저 고정하지 않습니다.
- 관측 제약: `E → D_rec`, `E → D_info`의 별도 decoder를 유지합니다.
  관측 pair의 W2 또는 signed measure가 공유 encoder를 학습시킵니다.
- Raw는 예측 손실만 사용하고 모델·seed마다 한 번만 학습합니다.
  latent의 손실 종류별로 같은 Raw 기준선을 재사용합니다.
- 기본 latent 두 설정은 `statistical+w2`, `statistical+signed_measure`입니다.
  `STATISTICAL_LOSSES="w2 kl_entropy signed_measure"`로 KL-entropy도 추가할 수 있습니다.
- static 손실과 두 flow 손실은 0입니다. 지형은 입력 정보로 남습니다.
  이 실험의 예측기는 결정론적이며 앙상블 확률 보정 성능을 검증하지 않습니다.
- 동일 데이터 분할·관측 origin·lead·seed·epoch/batch 설정을 사용합니다.
  calibration 분할에서 checkpoint를 선택하고 validation을 보고합니다.
  모든 학습을 끝내고 고정한 checkpoint를 별도 test 분할에 평가합니다.

현재 일별 전처리 정보에는 PINN에 필요한 같은 기압면의 U/V/T/Z/omega와 실제
표면기압 `sp` 전체가 포함되지 않으며, 24시간 PINN residual도 검증되지 않았습니다.
일별 PINN 실행은 거부합니다. 필요한 변수를 갖춘 **6시간 archive**에서는 아래처럼
`pinn_statistical`을 선택할 수 있습니다. 이는 PINN 단독 실험이 아니라 통계 제약에
PINN을 추가하는 실험입니다.

```bash
ARCHIVE=/lustre/home/yehyeon/<prepared-6hour>/surface.npz \
INFO=/lustre/home/yehyeon/<prepared-6hour>/information.npz \
CONSTRAINT_PAIRS="statistical pinn_statistical" \
bash scripts/run_raw_latent_comparison.sh
```

## 새 브랜치 받기부터 full 학습·test·그래프까지

할당받은 GPU 세션에서 실행합니다. 기본값은 **8개 M × (Raw 1개 + latent 손실 2개)
× seed 3개 = 72회 학습**, batch 16, 20 epochs입니다. `MAX_WINDOWS=0`, `MAX_CASES=0`은
각 분할을 잘라서 사용하는 제한을 끕니다. 테스트 결과를 보고 설정을 계속 바꾸는
단계에서는 `EVALUATE_TEST=0`으로 test를 남겨두세요.

원본은 `/lustre/home/mahmed/ERA5_0p25_DAILY`에서 읽기만 합니다. 코드·가상환경·전처리·
로그·출력·캐시는 `/lustre/home/yehyeon` 아래에 저장하며 기존 실험을 덮어쓰지 않습니다.
기본 출력 격자는 **16×32**, 6일 관측 → 1~5일 예측입니다. 원본 0.25° 해상도 실험이 아닙니다.

```bash
bash <<'BASH'
set -euo pipefail
export ERA5_ROOT=/lustre/home/mahmed/ERA5_0p25_DAILY
cd /lustre/home/yehyeon
export DAILY_WORK
DAILY_WORK=$(mktemp -d "$PWD/climate_manifold_models_XXXXXXXX")
mkdir -p "$DAILY_WORK"/{tmp,cache,logs}
export TMPDIR="$DAILY_WORK/tmp" XDG_CACHE_HOME="$DAILY_WORK/cache"
export TMP="$TMPDIR" TEMP="$TMPDIR" PIP_CACHE_DIR="$DAILY_WORK/cache/pip"
export TORCH_HOME="$DAILY_WORK/cache/torch" HF_HOME="$DAILY_WORK/cache/huggingface"
export CUDA_CACHE_PATH="$DAILY_WORK/cache/cuda" TRITON_CACHE_DIR="$DAILY_WORK/cache/triton"
export PYTHONPYCACHEPREFIX="$DAILY_WORK/cache/pycache" MPLCONFIGDIR="$DAILY_WORK/cache/matplotlib"
export NUMBA_CACHE_DIR="$DAILY_WORK/cache/numba" PYTHONNOUSERSITE=1

git clone --single-branch --branch feature/raw-latent-model-comparison \
  https://github.com/nayehyeon61-glitch/climate_manifold.git "$DAILY_WORK/code"
cd "$DAILY_WORK/code"
python3 -m venv "$DAILY_WORK/venv"
source "$DAILY_WORK/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'xarray==2024.11.0' -e '.[forecast,era5,test,plots]'
export PYTHON="$DAILY_WORK/venv/bin/python"
export DEVICE=cuda BATCH_SIZE=16 EPOCHS=20 SEEDS="7 19 43"
export MODELS="transformer mlp neural_ode climode convlstm simvp fourcastnet climax"
export STATISTICAL_LOSSES="w2 signed_measure" CONSTRAINT_PAIRS=statistical
export STATISTICAL_FLOW_WEIGHT=0 CONDITIONAL_FLOW_WEIGHT=0
export START_DATE=1979-01-01 END_DATE=2025-12-31
export TARGET_LAT_POINTS=16 TARGET_LON_POINTS=32
export MAX_WINDOWS=0 MAX_CASES=0 ORIGIN_STRIDE=1
export RUN_PREFLIGHT_TESTS=1 EVALUATE_TEST=1 MAKE_PLOTS=1 OMP_NUM_THREADS=4
export GPU_MAX_UTILIZATION=10 GPU_MAX_MEMORY_PERCENT=10 GPU_MIN_FREE_GIB=8
unset ARCHIVE INFO A_CHECKPOINT
git log -1 --oneline
printf 'Work directory: %s\n' "$DAILY_WORK"
bash scripts/run_raw_latent_comparison.sh
BASH
```

CUDA 12.8용 PyTorch는 서버 driver/GPU와 호환되어야 합니다. 실행기는 준비 전에 실제
CUDA kernel을 확인합니다. 각 학습/평가 시작 때 할당된 가시 GPU 중 사용률 ≤10%,
메모리 점유율 ≤10%, 여유 메모리 ≥8 GiB인 장치를 선택하고, 없으면 중단합니다.
이는 시작 시점의 선택 조건이며 학습 중 사용률을 10%로 제한하는 기능은 아닙니다.

기존 전처리를 재사용하려면 실행 전에 `ARCHIVE`와 `INFO`를 함께 기존 파일로 설정합니다.
위 블록의 `unset ARCHIVE INFO A_CHECKPOINT` 대신 두 경로와 `unset A_CHECKPOINT`를 사용하세요.
새 학습 결과는 여전히 새 `DAILY_WORK`에 저장됩니다. 같은 `DAILY_WORK`에서 재실행하면
전처리는 검증 후 재사용하고 학습 결과 폴더는 새로 만듭니다. optimizer resume는 아닙니다.

짧은 smoke 실행은 준비된 데이터에서 다음처럼 할 수 있습니다.

```bash
SEEDS=7 EPOCHS=1 MAX_WINDOWS=8 MAX_CASES=2 EVALUATE_TEST=0 \
bash scripts/run_raw_latent_comparison.sh
```

## 결과 확인

`$DAILY_WORK/runs/raw-latent-*/` 아래에 저장합니다.

| 파일 | 의미 |
|---|---|
| `source-commit.txt`, `run.log` | 실행 코드 버전과 로그 |
| `*.pt`, `*.metrics.json` | checkpoint와 학습·선택 분할 기록 |
| `*.validation.json`, `*.test.json` | 물리 단위 RMSE·ACC, origin/lead, 파라미터·입력 계약 |
| `comparison.validation.json`, `comparison.test.json` | 모델·손실별 seed 요약과 비교 |
| `comparison.*.raw-effects.csv` | 같은 M·seed의 Raw 대비 latent RMSE 감소·ACC 차이 |
| `comparison.*.statistical-effects.csv` | 같은 M·seed에서 W2/signed measure 교체 효과 |
| `plots/validation/`, `plots/test/` | 해당 분할의 PNG/PDF 그래프, 집계 CSV, 입력 hash manifest |

`MAKE_PLOTS=1`은 validation/test 비교가 끝날 때 각각 다음 그림을 자동 생성합니다.
기압 `msl`, 기온 `t2m`, 바람 `u10/v10`을 별도 변수로 표시합니다.

- `rmse_by_lead_<변수>.png/pdf`: 모델별 Raw/W2/signed measure RMSE 곡선.
- `acc_by_lead_<변수>.png/pdf`: 같은 비교의 ACC 곡선.
- `rmse_skill_percent_by_lead_<변수>.png/pdf`: 같은 seed의 Raw 대비 RMSE 개선율 곡선.
- `last_lead_rmse_<변수>.png/pdf`: 마지막 예측일의 모델·경로별 RMSE 막대그래프.
- `last_lead_improvement_heatmap.png/pdf`: 마지막 예측일의 Raw 대비 개선율. 양수가 개선입니다.
- `plot_summary.csv`: 그림에 사용한 seed 평균·표본 표준편차·유효 seed 수.

곡선/오차막대는 **seed 평균 ± 표본 표준편차**이며 신뢰구간이나 기상 앙상블의 불확실성이
아닙니다. 정의되지 않은 ACC나 0인 Raw RMSE의 개선율은 빈 구간/N/A로 표시합니다.
서로 다른 예측 실패 표본을 사용한 비교는 그림 생성을 거부합니다. 각 변수별 단위는
그대로 유지하며, 마지막 날을 선택해 더 좋아 보이는 리드를 고르는 절차는 없습니다.

학습을 다시 하지 않고 그림만 재생성하려면 기존 결과 폴더를 지정하세요.
기존 그림을 보존하기 위해 새 출력 폴더를 사용합니다.

```bash
RUN=/lustre/home/yehyeon/<실행폴더>/runs/<결과폴더>
python -m climate_manifold.downstream.plot_comparison \
  --comparison "$RUN/comparison.test.json" \
  --output "$RUN/plots/test-regenerated-$(date +%Y%m%d-%H%M%S)"
```

변수마다 단위가 다르므로 물리 RMSE를 합쳐 순위를 매기지 마세요. 변수·lead별로
Transformer의 Raw/latent 점수와 다른 M의 대응 점수를 비교합니다. 서로 다른 M의
파라미터 수·계산량·관측 이력 활용 방식 차이는 함께 보고합니다.

소프트웨어 검증은 합성 일별 NetCDF로 8개 M의 두 경로와 W2/signed measure를
실제 최적화하고 E/M/D gradient, 저장·복원 동일성, 미래 label을 입력하지 않는 예측,
validation/test 분리와 비교 CSV를 확인합니다. 서버 matrix는 모의 실행으로 72개 학습과
144개 평가 및 그래프 생성의 인자·순서를 검증합니다. 그래프 집계와 PNG/PDF 렌더링도
합성 결과로 확인합니다. 실제 ERA5 full 학습이나 GPU 성능 결과는 아닙니다.
