# 예측과 표현 제약을 분리한 쌍별 실험

이 브랜치의 새 실험은 하나의 encoder와 decoder를 공유하는 두 경로를 함께 학습합니다.
예측은 `E → F → D`, 표현 제약은 **관측 자료를 직접 복원하는 `E → D / D_I`**에서
계산합니다. 표현 제약 계산은 예측기 `F`를 호출하지 않습니다.

```mermaid
flowchart TD
    H["관측 history + origin 정보"] --> E["공유 encoder E"]
    O["관측 t−6h, t + 각 시점의 정보"] --> E
    E -->|"예측 경로"| F["선택한 공간 예측기 F"]
    F --> D["공유 기상장 decoder D"]
    E -->|"관측 복원 경로"| D
    E --> I["정보 decoder D_I"]
    D --> P["미래 출력: 예측 손실"]
    D --> R["관측 복원: 기본 복원 + 선택한 두 제약"]
    I --> R
```

`D`는 동일한 가중치로 관측 복원과 미래 기상장 출력을 담당합니다. `D_I`는 상층 변수와
지형 정보를 복원하는 보조 decoder입니다. 제약 경로를 위해 예측기나 encoder를 별도로 복제하지 않습니다.
PINN closure도 관측 복원 latent에서만 계산합니다. 이 학습의 추론 경로는 그대로
`관측 history → E → F → D → 미래 기상장`입니다.

## 정확히 두 개의 추가 제약

| `--constraint-pair` | PINN | Statistical | Static |
|---|---|---|---|
| `pinn_statistical` | 사용 | 사용 | 미사용 |
| `pinn_static` | 사용 | 미사용 | 사용 |
| `statistical_static` | 미사용 | 사용 | 사용 |

- **PINN:** 복원한 두 관측 시점의 기압면 운동량·온도·연속·층 두께 잔차와 closure 크기 규제.
  PDE 안의 시간차분은 유지하지만 기존의 별도 관측 tendency 감독과 지표 tendency 보조항은
  포함하지 않습니다. `HybridPINN.forward(include_tendency=False)`로 명시합니다.
- **Statistical:** 복원한 동적 지표 기상장 및 동적 상층 정보장의 면적 가중 공간 분위수 매칭.
  각 변수와 시점을 따로 비교합니다. 고정 지형을 포함하지 않으며, 앙상블 CRPS가 아닙니다.
- **Static:** 정보 decoder가 복원한 고정 변수의 면적 가중 L². 두 시점 모두 origin의 고정
  정보를 목표로 사용합니다. 현재 자료 계약에서는 지형 높이·경사가 해당합니다.
  위경도는 격자 좌표이며 별도의 학습 대상 변수로 추가되지 않습니다.

동적 자료의 기본 복원오차는 세 조합에 공통으로 **한 번** 유지합니다.

\[
L_{\rm rec}=\tfrac12\bigl(L_{\rm surface\ reconstruction}
                         +L_{\rm dynamic\ information\ reconstruction}\bigr).
\]

각 항은 train 자료로 정규화한 변수별 면적 가중 MSE를 평균합니다. Static 변수는 여기서
제외하여 Static 제약의 on/off 의미를 유지합니다. 이 공통 오차는 PINN+Static에서 상층
decoder가 물리 잔차만 작은 상수장을 출력하는 퇴화해를 견제합니다. Statistical 그룹은
이 점별 복원과 구별되는 **분포 매칭**입니다.

\[
L = L_{\rm forecast}+\lambda_\Delta L_{\rm future\ tendency}
    +\lambda_{\rm rec}L_{\rm rec}
    +\sum_{g\in\text{선택한 두 그룹}}\lambda_g L_g.
\]

기본 외부 가중치는 reconstruction 0.1, statistical 0.1, static 0.05, PINN 0.1입니다.
미래 tendency는 기존 예측 손실의 설정을 사용합니다. 예측장 surface physics, 미래 information
MSE, 미래 분포·static·PINN은 이 경로에서 추가로 부과하지 않습니다.

## 자료와 gradient 계약

기본 예측 입력은 24시간 간격의 6개 관측 지표장과 origin 정보이며, 출력은 6시간 간격의
20개 미래장입니다. 제약 경로는 같은 관측 구간 안의 **origin−6h와 origin** 두 장만 가져와
각 시점에 정확히 대응하는 정보 자료와 함께 복원합니다. 예측 입력의 24시간 간격을
6시간이라고 가정하지 않습니다. 두 장이 모두 관측 구간에 있어야 하므로 history span은
최소 2개의 원자료 시점이어야 합니다.

`constraint_states`, `constraint_information`, `constraint_dt_hours`는 제약 손실에만
전달됩니다. 미래 상층 정보는 제약 경로에서 읽지 않습니다. train/calibration/validation/test
분할과 train-only 정규화는 기존 자료 계약을 유지합니다.

제약 손실만 역전파하면 `F`에는 gradient가 없고, 공유 `E`, `D`, `D_I` 및 활성 closure에
전달됩니다. 미래 예측 손실은 `E`, `F`, `D`를 함께 갱신합니다. 공유 encoder/decoder를 통해
제약이 예측에 간접 영향을 주므로 두 학습 목적이 완전히 독립이라는 의미는 아닙니다.

## 전체 세 조합 실행

직접 예측 비교군 `data → F → future fields`도 함께 학습합니다. 이 비교군은
E/D 없이 원본 격자에서 같은 계열의 예측기를 사용하며, 같은 origin 상층/지형 정보를
입력으로 받습니다. 예측·변화량 손실만 사용하고 복원·PINN·Statistical·Static 손실은
사용하지 않습니다. Raw ClimODE는 같은 transport core를 원본 격자에 적용한 matched
adaptation이며, 별도의 vendor Gaussian ClimODE 실행이 아닙니다.

| 실험군 | Neural ODE | ClimODE |
|---|---|---|
| Raw 직접 예측 | 1회 | 1회 |
| E→F→D + PINN·Statistical | 1회 | 1회 |
| E→F→D + PINN·Static | 1회 | 1회 |
| E→F→D + Statistical·Static | 1회 | 1회 |

따라서 seed당 8회이며, Raw 비교군은 제약 조합마다 반복하지 않고 예측기·seed별로
한 번만 학습합니다. `INCLUDE_RAW=0`이면 직접 예측 비교군을 제외합니다.

저장소 루트에서 다음을 실행합니다. INFO에는 같은 기압면의 U/V/T/Z/omega와 실제 지표기압
sp가 필요하며, 두 PINN 조합에서는 모듈이 자동으로 활성화됩니다. Statistical+Static은
PINN 모듈 없이 실행합니다.

```bash
export ARCHIVE=/absolute/path/to/surface.npz
export INFO=/absolute/path/to/information_pinn
export RUN=runs/split_pairs_001
export DEVICE=cuda MODELS="neural_ode climode" SEEDS=7
export INCLUDE_RAW=1 PAIRS="pinn_statistical pinn_static statistical_static"
export EPOCHS=20 BATCH_SIZE=16
bash scripts/run_pairwise_manifold_comparison.sh
```

위 명령은 총 8회 학습합니다. `SEEDS`를 생략하면 기본 3개 seed(7, 19, 43)로 총 24회입니다.
각 실행은 공통 calibration 미래 MSE로 checkpoint를 선택하고 같은 물리장 평가를 사용합니다.
PINN+Statistical만 먼저 실행하려면 `PAIRS=pinn_statistical`로 바꿉니다. 이때는
2개 예측기 × (Raw + PINN·Statistical) × 1개 seed로 총 4회입니다.
`MODELS=mlp`, `MODELS=convlstm`, `MODELS=simvp`도 지원하지만 기본 예측기 2종에는 포함하지 않습니다.

## 다섯 예측기의 PINN+Statistical + Raw 비교

`mlp`, `neural_ode`, `climode`, `convlstm`, `simvp` 모두 동일한 자료 계약에서
`data → F → future fields`와 `E → F → D`를 비교할 수 있습니다. ConvLSTM과 SimVP-gSTA는
관측/예측 시간 정보를 조건으로 받는 공간 예측기이며, **joint spatial + matched Raw**만
지원합니다. 원본 논문 checkpoint 대신 이 프로젝트의 자료·출력 간격에 맞춘 adaptation을
새로 학습합니다. [예측기 설명](downstream.md#선택-가능한-예측기)과
[출처·수정 내역](../THIRD_PARTY_NOTICES.md)을 참고하세요.

기존 Python 학습 환경을 활성화하고 저장소 루트에서 실행합니다. ARCHIVE와 INFO의 두 경로를
실제 자료 경로로 변경하세요. INFO에는 PINN용 변수와 실제 지표기압 `sp`가 필요합니다.

```bash
(
set -euo pipefail

git fetch origin
git switch feature/split-manifold-pairwise-constraints
git pull --ff-only origin feature/split-manifold-pairwise-constraints
python -m pip install -e '.[forecast]'

export ARCHIVE="/absolute/path/to/surface.npz"
export INFO="/absolute/path/to/pinn_information_shards"
export RUN="runs/pinn_statistical_five_models_$(date +%Y%m%d_%H%M%S)"

export MODELS="mlp neural_ode climode convlstm simvp"
export PAIRS="pinn_statistical" INCLUDE_RAW=1 SEEDS=7
export TRAINING_MODE=joint INITIALIZATION=fresh LATENT_LAYOUT=spatial ANCHOR=none
unset A_CHECKPOINT CLIMODE_REFERENCE_DIR

export LATENT_CHANNELS=32 SPATIAL_DOWNSAMPLE=2 SPATIAL_HIDDEN_DIM=64 HIDDEN_DIM=128
export HISTORY_STEPS=6 HISTORY_STRIDE=4 HORIZON_STEPS=20
export PINN_LEVELS="500 850" PINN_WEIGHT=0.1 STATISTICAL_WEIGHT=0.1 STATIC_WEIGHT=0
export RECONSTRUCTION_WEIGHT=0.1 TENDENCY_WEIGHT=0.1
export PYTHON=python DEVICE=cuda EPOCHS=20 BATCH_SIZE=16 LEARNING_RATE=0.001
export WINDOW_STRIDE=4 MAX_WINDOWS=0 MAX_CASES=0 ORIGIN_STRIDE=1

mkdir -p logs
bash scripts/run_pairwise_manifold_comparison.sh \
  2>&1 | tee "logs/${RUN##*/}.log"
)
```

과거 6개 관측(24시간 간격)에서 미래 20개 장(6시간 간격, 120시간)을 예측합니다.
Raw와 E→F→D는 같은 history와 origin 정보에 접근하고, 미래 정답을 입력으로 받지 않습니다.
모델별 latent 표현의 학습과 예측기 학습은 함께 진행합니다. Raw는 manifold 보조 손실을
사용하지 않습니다. 모델별 용량과 연산량은 다르므로 parameter 수·실행 비용도 함께 보고합니다.

| 선택 | seed당 학습 횟수 |
|---|---|
| 5개 모델 + PINN·Statistical + Raw | 10회 |
| `MODELS="convlstm simvp"` + PINN·Statistical + Raw | 4회 |
| 5개 모델 + 세 제약 조합 + Raw | 20회 |

세 조합을 모두 실행하려면 `PAIRS="pinn_statistical pinn_static statistical_static"`로 바꾸고
`STATIC_WEIGHT=0.05`를 설정합니다. `SEEDS="7 19 43"`이면 표의 횟수의 3배입니다.
Raw는 항상 모델·seed별 한 번만 학습합니다. 각 모델의 checkpoint·validation 점수와
`comparison.json`, Raw 대비 `comparison.raw-effects.csv`를 새 RUN 폴더에서 확인할 수 있습니다.

## 단일 조합과 기존 경로

단일 조합은 다음과 같이 실행합니다.

```bash
python -m climate_manifold.downstream.train \
  --archive "$ARCHIVE" --information "$INFO" \
  --training-mode joint --initialization fresh \
  --bridge latent --model neural_ode --latent-layout spatial \
  --constraint-pair pinn_statistical \
  --reconstruction-weight 0.1 --statistical-weight 0.1 --pinn-weight 0.1 \
  --epochs 20 --batch-size 16 --device cuda \
  --output runs/split_single/model.pt
```

새 경로는 `--constraint-pair`로 선택합니다. 지정하지 않은 기존 학습 명령과
`run_model_comparison.sh`는 예측 궤적에 보조 제약을 적용하던 기존 실험을 재현합니다.
새 쌍별 runner에는 기존 `INFORMATION_WEIGHT`, `PHYSICS_WEIGHT`, `DISTRIBUTION_WEIGHT`
설정이 필요하지 않습니다. 실제 활성 가중치와 경로는 checkpoint와 평가 보고서에 기록됩니다.

## 결과 해석

세 조합은 같은 자료·예측기·seed·학습 예산에서 비교합니다. 예를 들어 PINN+Static과
Statistical+Static은 Static을 공유하면서 다른 제약을 교체하는 비교입니다. 한 그룹만
추가한 실험은 아니므로 그 차이를 PINN 하나의 인과적 효과로 해석하지 않습니다.

보고서는 조합별로 seed를 집계하며 서로 다른 조합을 한 모델의 반복 실행처럼 합치지
않습니다. 변수·lead별 물리 단위 RMSE/ACC, 장기 rollout의 유한성, 계산 비용을 평가합니다.
각 제약 조합과 같은 예측기·seed의 Raw 비교는 `comparison.raw-effects.csv`에,
제약 조합 간 비교는 `comparison.constraint-effects.csv`에 기록합니다. Raw와 E→F→D는
표현·용량·추가 감독이 함께 달라지므로 이는 전체 모델 비교이며 제약만의 효과는 아닙니다.
학습/검증 오차의 차이와 여러 seed의 결과를 확인해야 과적합 변화에 대해 판단할 수 있습니다.
경로 분리나 손실 항 개수 감소 자체는 일반화 개선을 보장하지 않습니다.

이 설계는 **관측된 상태들의 표현**에 물리 제약을 주며, 생성된 미래 궤적의 PDE 만족을
직접 감독하지 않습니다. 특히 관측 복원 PINN 잔차가 작다는 사실만으로 미래 궤적의 물리적
타당성을 주장할 수 없습니다.

## 구현 검증

전체 테스트 410개를 통과했습니다. 제약만 역전파했을 때 예측기 F의 gradient가 없는지,
미래 information을 바꾸어도 관측 제약이 변하지 않는지, 세 조합의 활성 항과 checkpoint
재로딩이 올바른지를 확인했습니다. 추가로 합성 자료에서 Neural ODE·ClimODE × (Raw + 세 조합)을
batch 16으로 각각 1 epoch 학습하고 120시간 예측·평가·집계를 완료했습니다. 8개 실험 모두
동일한 평가 origin에서 유한한 출력을 생성했으며, Raw 대비 6개 비교와 제약 조합 간
6개 비교를 별도로 기록했습니다. 이는 소프트웨어 동작 검증이며,
실제 기상 예측력이나 과적합 감소의 근거는 아닙니다.

ConvLSTM·SimVP 추가 후에도 각 모델의 Raw/PINN+Statistical 경로를 batch 16,
history 6장×24시간, horizon 20장×6시간으로 각각 1 epoch 학습했습니다.
4개 실행 모두 checkpoint 재로딩·평가·비교를 완료했고, 동일한 두 평가 origin에서
120시간까지 유한한 출력을 생성했습니다. 이 검사는 작은 4×8 합성 격자와
hidden width 8/latent channels 2의 CPU 실행입니다. 실제 ERA5 예측력과
기본 width 128 설정의 GPU 메모리 적합성은 별도 실험이 필요합니다.

## 코드 위치

| 역할 | 코드 |
|---|---|
| 관측 쌍 구성·학습 분기·메타데이터 | `src/climate_manifold/downstream/train.py` |
| E→D 제약 계산·쌍별 활성화 | `src/climate_manifold/downstream/reconstruction_objective.py` |
| 물리 잔차와 선택적 tendency 감독 | `src/climate_manifold/hybrid_pinn.py` |
| 예측 E→F→D | `src/climate_manifold/downstream/pipeline.py` |
| 전체 쌍별 실행 | `scripts/run_pairwise_manifold_comparison.sh` |
| 평가 및 조합별 집계 | `src/climate_manifold/downstream/evaluate.py`, `compare.py` |
