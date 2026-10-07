# 전체 공간·동역학 Manifold 비교

브랜치 `feature/full-manifold-matrix`. 기본 **12조건 × 3통계 손실 × 8모델 × 3seed =
864회**입니다. Raw도 각 손실 suite에서 같은 설정으로 재학습합니다.

| 예측 경로 | Dynamics 보조 감독 | 조건 수 |
|---|---|---:|
| Raw → M | 없음 | 1 |
| E_s → M → D_forecast | 공간만 | 1 |
| T2(Raw, E_s) → M | 공간만 | 1 |
| (E_s + P(E_d)) → M → D_forecast | field / vector_only / field_latent | 3 |
| T2(Raw, E_s, E_d) → M — T1 OFF | 같은 세 감독 | 3 |
| T2(Raw, T1(E_s, E_d)) → M — T1 ON | 같은 세 감독 | 3 |

**Dynamics guide도 있습니다.** 관측 history에서 E_d가 만든 temporal latent를 사용합니다.
T1은 공간·동역학 guide를 상호작용시키고, T2는 Raw history와 guide를 결합합니다.
T1 OFF에도 Dynamics guide와 T2는 유지됩니다. Raw/Latent/공간만에는 두 guide의 T1
비교를 적용하지 않습니다. 예측 M의 `transformer` 선택지는 T1/T2와 별도 모듈입니다.

`field`는 보조 F_d→D_d의 미래 장·통계 손실, `vector_only`는 미래를 E_d로 encode한
상태 latent의 일치 손실, `field_latent`는 둘 다입니다. vector는 속도 벡터장이나
Flow Matching을 뜻하지 않습니다. vector_only에서 D_d는 동결되고 보조 미래 장 손실은
0입니다. 모든 모드에서 주예측 M의 미래 장 손실과 공간 E_s의 관측 복원·통계 손실은
유지됩니다. 표현과 M은 공동 학습하며 미래 정답은 감독에만 사용합니다.

W2/KL-entropy/signed measure는 공간 통계와 Dynamics 미래 장 통계에 함께 적용합니다.
vector_only에서는 후자를 계산하지 않습니다. 손실 suite 비교는 통계 감독 전체의 선택이며
공간 손실만의 효과를 분리하지 않습니다. 세 손실이 guide의 확률 샘플링을 추가하지는
않습니다. PINN/static/두 Flow 손실은 이번 행렬에서 사용하지 않습니다.

## 브랜치 받기 → 학습 → test → 그래프

할당받은 GPU 세션에서 아래를 한 번 실행합니다. 모든 쓰기는 `/lustre/home/yehyeon`
아래에, ERA5 원본은 지정 위치에서 읽기만 합니다. 아래는 1979–2025 자료를 준비하여
**16×32 격자, 6일 관측 → 미래 1–5일, msl/t2m/u10/v10**을 예측합니다.
0.25도 전체 격자와 13기압면 변수를 전부 예측하는 설정이 아닙니다.

```bash
bash <<'BASH'
set -euo pipefail
cd /lustre/home/yehyeon
export DAILY_WORK
DAILY_WORK=$(mktemp -d "$PWD/climate_manifold_full_XXXXXXXX")
export ERA5_ROOT=/lustre/home/mahmed/ERA5_0p25_DAILY
mkdir -p "$DAILY_WORK"/{tmp,cache,logs}
export TMPDIR="$DAILY_WORK/tmp" TMP="$DAILY_WORK/tmp" TEMP="$DAILY_WORK/tmp"
export XDG_CACHE_HOME="$DAILY_WORK/cache" PIP_CACHE_DIR="$DAILY_WORK/cache/pip"
export TORCH_HOME="$DAILY_WORK/cache/torch" HF_HOME="$DAILY_WORK/cache/huggingface"
export CUDA_CACHE_PATH="$DAILY_WORK/cache/cuda" TRITON_CACHE_DIR="$DAILY_WORK/cache/triton"
export PYTHONPYCACHEPREFIX="$DAILY_WORK/cache/pycache" MPLCONFIGDIR="$DAILY_WORK/cache/matplotlib"
export NUMBA_CACHE_DIR="$DAILY_WORK/cache/numba" PYTHONNOUSERSITE=1

git clone --single-branch --branch feature/full-manifold-matrix \
  https://github.com/nayehyeon61-glitch/climate_manifold.git "$DAILY_WORK/code"
cd "$DAILY_WORK/code"
python3 -m venv "$DAILY_WORK/venv"
source "$DAILY_WORK/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'xarray==2024.11.0' -e '.[forecast,era5,test,plots]'

export PYTHON="$DAILY_WORK/venv/bin/python" DEVICE=cuda
export FULL_MATRIX_LOSSES="w2 kl_entropy signed_measure"
export MODELS="mlp neural_ode climode convlstm simvp fourcastnet climax transformer"
export SEEDS="7 19 43" EPOCHS=20 BATCH_SIZE=16
export DYNAMICS_ENCODER=conv_gru DYNAMICS_HIDDEN_DIM=32 DYNAMICS_CHANNELS=8
export DYNAMICS_LATENT_WEIGHT=0.1
export DYNAMICS_RECONSTRUCTION_WEIGHT=0.1 DYNAMICS_STATISTICAL_WEIGHT=0.1 DYNAMICS_TENDENCY_WEIGHT=0
export GUIDE_INTERACTION_DEPTH=1 GUIDE_FUSION_DEPTH=2
export VARIABLE_CONDITIONING=0 GUIDE_DIRECT_INFORMATION=1
export START_DATE=1979-01-01 END_DATE=2025-12-31
export TARGET_LAT_POINTS=16 TARGET_LON_POINTS=32
export HISTORY_STEPS=6 HORIZON_STEPS=5 MAX_WINDOWS=0 MAX_CASES=0 ORIGIN_STRIDE=1
export RUN_PREFLIGHT_TESTS=1 EVALUATE_TEST=1 MAKE_PLOTS=1 OMP_NUM_THREADS=4
export GPU_MAX_UTILIZATION=10 GPU_MAX_MEMORY_PERCENT=10 GPU_MIN_FREE_GIB=8 GPU_WAIT_SECONDS=0
unset ARCHIVE INFO A_CHECKPOINT RUN RESUME AUX_ROLLOUT_STEPS
printf 'Work directory: %s\n' "$DAILY_WORK"
bash scripts/run_full_manifold_matrix.sh
BASH
```

GPU guard는 다른 compute process가 없고 사용률·메모리 점유율 각각 10% 이하,
여유 8 GiB 이상인 허용 장치를 선택합니다. **시작 조건이며 학습 중 상한은 아닙니다.**
조건에 맞는 장치가 없으면 즉시 종료합니다. CUDA 설치에는 cu128을 지원하는 서버
드라이버가 필요합니다. 모든 fit/eval은 순차 실행합니다.

8모델은 저장소의 동일 격자·변수 계약에 맞춘 구현/적응 모델입니다. 공식 pretrained
최고 성능 재현과 구분해야 합니다. GraphCast 공식 경로는 이 행렬에 포함하지 않습니다.

## 결과

| DAILY_WORK 아래 경로 | 내용 |
|---|---|
| `runs/full-w2/` | W2 288회 |
| `runs/full-kl_entropy/` | KL-entropy 288회 |
| `runs/full-signed_measure/` | signed measure 288회 |
| 각 suite의 `comparison.json`, `comparison.test.json` | validation/test 비교 |
| 각 suite의 `plots/validation/`, `plots/test/` | RMSE·ACC·비교 효과 PNG/PDF |
| `full-matrix-summary.csv` | 손실·모델·경로·seed·split별 집계 점수/용량/시간 |
| `logs/` | 준비·테스트·학습 로그 |

T1 ON/OFF는 각각 독립 학습합니다. 같은 checkpoint의 T1 우회 진단은 끕니다.
field↔hybrid, vector_only↔field/hybrid, Latent↔Guided, T1 OFF↔ON 효과표도 생성됩니다.
Dynamics↔공간만은 전체 RMSE/ACC 및 Raw 대비 표에서 비교합니다. 기존 `dynamics_effects`
표는 legacy direct 전용이므로 이번 rollout 행렬에서 비어 있을 수 있습니다.

Raw 대비에는 모델 용량 증가 효과도 포함됩니다. 통계 guide 자체의 기여를 분리하려면
별도 zero-guide/보조 손실 제거 비교가 필요합니다. 실험별 latent 좌표·스케일이 다를 수
있으므로 latent 오차만으로 우열을 정하지 않습니다. vector_only의 동결 D_d는 물리장
평가에서 제외하고, 공통 주예측 M의 미래 물리장 점수를 비교합니다.

## 재개·선택

처음 출력된 DAILY_WORK를 보관하세요. **원래 DAILY_WORK와 동일한 환경변수·코드**를
다시 설정하고 `RESUME=1 bash scripts/run_full_manifold_matrix.sh`로 재개합니다.
새 전체 명령은 새 작업 폴더를 만듭니다. 데이터·설정·소스 변경 시 재사용을 거절합니다.
첫 suite의 테스트만 실행하고 다음 손실은 동일 준비 자료를 검증·재사용합니다.
새 DAILY_WORK마다 전처리를 다시 합니다.

`FULL_MATRIX_LOSSES="signed_measure"`, `MODELS="mlp neural_ode"`, `SEEDS="7"`처럼
줄일 수 있습니다. 기본은 3손실/8모델/3seed, batch 16, 20 epochs입니다.
공통 평가 origin을 유지하기 위해 이 전체 행렬에서 AUX_ROLLOUT_STEPS를 늘리지 않습니다.

기존 실행기의 기본 행렬은 유지합니다. 새 wrapper만 `INCLUDE_LEGACY_DYNAMICS=0`으로
이전 direct Dynamics 3조건을 제외합니다. 직접 이 옵션을 끌 때는
`INCLUDE_DYNAMICS_CONSISTENCY=1`, `INCLUDE_ZERO_DYNAMICS=0`,
`INCLUDE_TWO_STAGE_BYPASS=0`도 필요합니다.
