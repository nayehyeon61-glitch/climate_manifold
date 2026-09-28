# 예측과 표현 제약을 분리한 쌍별 실험

이 브랜치의 새 실험은 하나의 encoder를 공유하는 두 경로를 함께 학습합니다.
예측은 `E → F → D`, 표현 제약은 **관측 기상장의 `E → D_rec`와 관측 정보의 `E → D_I`**에서
계산합니다. D_rec는 예측 decoder D와 독립된 장복원 decoder입니다. 기존 관측 제약 계산은
예측기 `F`와 예측 decoder `D`를 호출하지 않습니다. 기본값이 꺼진 선택적
[예측 분포 flow 손실](#선택적-현재미래-분포-flow-손실)은 별도로 F의 미래 latent에서 계산합니다.
새 [관측 조건부 Flow Matching](#관측-쌍-조건부-flow-matching)은 관측 쌍과 현재 보조 복원에서
조건을 만들고 실제 미래 분포로 향하는 별도 벡터장을 학습합니다. 두 옵션 모두 기본값은 꺼짐이며,
한 실험에서 두 가지 Flow 항을 동시에 활성화하지 않습니다.

```mermaid
flowchart TD
    H["관측 history + origin 정보"] --> E["공유 encoder E"]
    O["관측 t−6h, t + 각 시점의 정보"] --> E
    E -->|"예측 경로"| F["선택한 공간 예측기 F"]
    F --> D["기상장 decoder D"]
    E -->|"관측 기상장 복원 경로"| DR["별도 장복원 decoder D_rec"]
    E -->|"관측 정보 복원 경로"| I["정보 decoder D_I"]
    D --> P["미래 출력: 예측 손실"]
    DR --> SR["현재 기상장: 복원 + 선택 시 W₂²"]
    I --> R["동적 정보 복원 + 선택한 두 제약"]
```

기본 `--constraint-decoder separate_surface_and_information`에서는 관측 기상장 복원을
독립된 D_rec에 맡깁니다. D_rec는 D와 같은 구조·초기 가중치로 시작하지만 파라미터를 공유하지
않습니다. **D 자체를 동결하는 것은 아닙니다.** D는 미래 기상장 예측 손실로 계속 학습합니다.
`D_I`는 상층 변수와 지형 정보를 복원하는 보조 decoder입니다. 예측기나 encoder를 별도로
복제하지 않습니다. D_rec는 새 분리 모드에만 추가되며 Raw와 기존 모드의 구조는 유지합니다.
PINN closure도 관측 복원 latent에서만 계산합니다. 이 학습의 추론 경로는 그대로
`관측 history → E → F → D → 미래 기상장`입니다.

기존 경로도 삭제하지 않습니다. 아래 옵션은 `--constraint-pair`를 사용하는 latent 공동 학습에만 적용합니다.

| `--constraint-decoder` | 관측 기상장 복원 | 관측 정보 복원 | 제약이 예측 D를 직접 갱신 |
|---|---|---|---|
| `separate_surface_and_information` (기본) | 별도 D_rec | D_I | 아니요 |
| `information_only` | 사용 안 함 | D_I | 아니요 |
| `surface_and_information` | 예측 D 공유 | D_I | 예 |

## 정확히 두 개의 추가 제약

| `--constraint-pair` | PINN | Statistical | Static |
|---|---|---|---|
| `pinn_statistical` | 사용 | 사용 | 미사용 |
| `pinn_static` | 사용 | 미사용 | 사용 |
| `statistical_static` | 미사용 | 사용 | 사용 |

- **PINN:** 복원한 두 관측 시점의 기압면 운동량·온도·연속·층 두께 잔차와 closure 크기 규제.
  PDE 안의 시간차분은 유지하지만 기존의 별도 관측 tendency 감독과 지표 tendency 보조항은
  포함하지 않습니다. `HybridPINN.forward(include_tendency=False)`로 명시합니다.
- **Statistical:** 기본값에서는 D_rec의 관측 기상장과 D_I의 동적 정보장에 면적 가중 공간 분위수 W₂² 매칭.
  각 변수와 시점을 따로 비교합니다. 고정 지형을 포함하지 않으며, 앙상블 CRPS가 아닙니다.
  **해면기압 `msl`을 포함한 surface archive의 모든 변수**가 기상장 분포 손실에 포함됩니다.
  해면기압에만 걸리는 손실은 아닙니다. `information_only`에서는 동적 정보 W₂²만 사용하므로
  기상장 W₂²와 `msl` 분포 손실이 꺼집니다. D_I에는 `msl` 출력을 추가하지 않습니다.
  이전 `surface_and_information`에서는 같은 기상장 손실을 예측 D로 계산합니다.
- **Static:** 정보 decoder가 복원한 고정 변수의 면적 가중 L². 두 시점 모두 origin의 고정
  정보를 목표로 사용합니다. 현재 자료 계약에서는 지형 높이·경사가 해당합니다.
  위경도는 격자 좌표이며 별도의 학습 대상 변수로 추가되지 않습니다.

기상장과 동적 정보의 기본 복원오차는 세 조합에 공통으로 **한 번씩** 유지합니다.

\[
L_{\rm rec}=\tfrac12\left(L_{\rm surface\ reconstruction}^{D_{\rm rec}}
  +L_{\rm dynamic\ information\ reconstruction}^{D_I}\right),
\qquad
L_{\rm statistical}=\tfrac12\left(L_{\rm surface\ W_2^2}^{D_{\rm rec}}
  +L_{\rm dynamic\ information\ W_2^2}^{D_I}\right).
\]

복원 항은 train 자료로 정규화한 변수별 면적 가중 MSE를 평균합니다. Static 변수는 여기서
제외하여 Static 제약의 on/off 의미를 유지합니다. 이 공통 오차는 PINN+Static에서 상층
decoder가 물리 잔차만 작은 상수장을 출력하는 퇴화해를 견제합니다. Statistical 그룹은
이 점별 복원과 구별되는 **분포 매칭**입니다.

기본값과 이전 `surface_and_information`은 reconstruction과 Statistical 각각에 같은
`0.5 × (기상장 손실 + 동적 정보 손실)`을 사용하고, 기상장 손실을 받는 decoder가 다릅니다.
`information_only`는 제거한 기상장 항을 0으로 두고 평균하는 대신 정보 손실 자체를 사용합니다.
따라서 외부 가중치가 같아도 정보 전용 모드는 정보 항의 실효 가중치가 다릅니다.
실험 비교 시 decoder 모드와 가중치를 함께 기록합니다.

\[
L = L_{\rm forecast}+\lambda_\Delta L_{\rm future\ tendency}
    +\lambda_{\rm rec}L_{\rm rec}
    +\sum_{g\in\text{선택한 두 그룹}}\lambda_g L_g.
\]

기본 외부 가중치는 reconstruction 0.1, statistical 0.1, static 0.05, PINN 0.1입니다.
미래 tendency는 기존 예측 손실의 설정을 사용합니다. 기본값에서는 예측장 surface physics,
미래 information MSE, 미래 분포·static·PINN을 추가로 부과하지 않습니다.
선택적 Flow를 활성화하면 아래에 정의한 예측 분포 변화율 손실 또는 관측 조건부
Flow Matching 손실을 별도로 추가합니다. 두 방식의 직접 gradient 경로는 다릅니다.

## 자료와 gradient 계약

기본 예측 입력은 24시간 간격의 6개 관측 지표장과 origin 정보이며, 출력은 6시간 간격의
20개 미래장입니다. 제약 경로는 같은 관측 구간 안의 **origin−6h와 origin** 두 장만 가져와
각 시점에 정확히 대응하는 정보 자료와 함께 복원합니다. 예측 입력의 24시간 간격을
6시간이라고 가정하지 않습니다. 두 장이 모두 관측 구간에 있어야 하므로 history span은
최소 2개의 원자료 시점이어야 합니다.

`constraint_states`, `constraint_information`, `constraint_dt_hours`는 제약 손실에만
전달됩니다. 미래 상층 정보는 기존 관측 제약 경로에서 읽지 않습니다. 선택적 flow가 켜진 경우에만
미래 동적 정보가 flow 손실의 정답으로 사용되며 encoder나 예측기 입력에는 들어가지 않습니다.
train/calibration/validation/test
분할과 train-only 정규화는 기존 자료 계약을 유지합니다.

기본 분리 모드에서 기존 관측 제약만 역전파하면 `F`와 예측 `D`에는 gradient가 없고,
공유 `E`, `D_rec`, `D_I` 및 활성 closure에 전달됩니다. 미래 예측 손실은 `E`, `F`, `D`를 함께
갱신하며 D_rec를 직접 갱신하지 않습니다. 정보 전용 모드에는 D_rec가 없으며,
이전 공유 모드에서는 제약 손실도 D를 갱신합니다. 공유 encoder를 통해
제약이 예측에 간접 영향을 주므로 두 학습 목적이 완전히 독립이라는 의미는 아닙니다.
별도 D_rec를 추가하면 학습 파라미터 수가 늘어납니다. decoder 간 직접적인 손실 충돌을
분리하는 설계이며, 중복 정보나 과적합을 제거했다는 증거는 아닙니다.

장복원 출력은 `pipeline.reconstruction_decoder(raw_latent)`로 계산하고,
학습 로그의 `reconstruction_surface`와 `statistical_surface`에서 해당 손실을 확인합니다.
평가의 no-grad 현재장 복원·latent 진단은 기존 예측 D를 진단하는 항목으로 유지합니다.
이 진단값은 D_rec 성능을 측정한 값이 아니며, 학습 손실에도 추가되지 않습니다.

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
export CONSTRAINT_DECODER=separate_surface_and_information
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
export CONSTRAINT_DECODER=separate_surface_and_information
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
  --constraint-decoder separate_surface_and_information \
  --reconstruction-weight 0.1 --statistical-weight 0.1 --pinn-weight 0.1 \
  --epochs 20 --batch-size 16 --device cuda \
  --output runs/split_single/model.pt
```

새 경로는 `--constraint-pair`로 선택합니다. 지정하지 않은 기존 학습 명령과
`run_model_comparison.sh`는 예측 궤적에 보조 제약을 적용하던 기존 실험을 재현합니다.
새 쌍별 runner에는 기존 `INFORMATION_WEIGHT`, `PHYSICS_WEIGHT`, `DISTRIBUTION_WEIGHT`
설정이 필요하지 않습니다. 실제 활성 가중치와 경로는 checkpoint와 평가 보고서에 기록됩니다.

새 명령에서 `--constraint-decoder`를 생략하면 `separate_surface_and_information`이 적용됩니다.
기존의 예측 D 공유 제약을 재현하려면 새 RUN에서 다음처럼 실행합니다. Raw 비교군에는 이 옵션을
전달하지 않으며, 직접 예측 경로는 그대로 유지됩니다.

```bash
export CONSTRAINT_DECODER=surface_and_information
export RUN="runs/split_pairs_both_decoders_$(date +%Y%m%d_%H%M%S)"
bash scripts/run_pairwise_manifold_comparison.sh
```

다시 정보 decoder만 사용하려면 `CONSTRAINT_DECODER=information_only`로 설정합니다.
기존 checkpoint의 손실 의미와 모델 구조를 새 기본값으로 바꾸지는 않습니다. 이전 v1 제약 메타데이터는
예측 D 공유 모드로 해석하며, 서로 다른 decoder 모드의 보고서를 같은 쌍별 비교에 섞지 않습니다.
새 분리 모드의 D_rec는 checkpoint에 함께 저장합니다. 기존 checkpoint를 재평가하는 것만으로
독립된 장복원 decoder가 생기거나 재학습되지는 않습니다.

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
경로 분리 자체는 일반화 개선을 보장하지 않습니다. 새 장복원 decoder의 추가 용량도 함께 고려합니다.

PINN은 **관측된 상태들의 표현**에 물리 제약을 주며, 생성된 미래 궤적의 PDE 만족을
직접 감독하지 않습니다. 특히 관측 복원 PINN 잔차가 작다는 사실만으로 미래 궤적의 물리적
타당성을 주장할 수 없습니다.

## 구현 검증

별도 장복원 decoder 추가 후 전체 테스트 **526개**를 통과했습니다. D_rec가 예측 D와
파라미터를 공유하지 않는지, 관측 제약은 D_rec를 갱신하면서 예측 F/D에는 직접 gradient를
전달하지 않는지, 미래 예측 손실은 E/F/D를 계속 갱신하는지 확인했습니다. 기상장 복원·분포
손실의 복구, 세 decoder 모드의 선택, checkpoint 재로딩과 이전 모드 호환성, Raw 비교군도
검증했습니다. 이 결과는 소프트웨어 동작 검증이며 실제 예측력이나 과적합 감소를 뜻하지 않습니다.

이전 정보 decoder 전용 모드 추가 시 전체 테스트 **460개**를 통과했습니다. 해당 모드에서
관측 제약과 학습 루프가 기상장 D를 호출하지 않는지, 예측 손실은 E/F/D를 계속 갱신하는지,
정보 손실의 가중치·기존 모드 복구·checkpoint 재로딩·Raw 비교군이 올바른지 확인했습니다.
평가의 no-grad 현재장 복원 진단은 유지하며, 학습 손실에는 포함하지 않습니다.
새 decoder 범위는 checkpoint와 평가 기록에 저장하고, 서로 다른 범위를 같은 쌍별 실험으로
집계하지 않습니다. 실제 기상자료에서의 예측력·과적합 감소는 아직 검증하지 않았습니다.

정보 전용 모드 추가 전 전체 테스트 410개를 통과했습니다. 제약만 역전파했을 때 예측기 F의 gradient가 없는지,
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
| E→D_rec / E→D_I 제약 계산·decoder 모드·쌍별 활성화 | `src/climate_manifold/downstream/reconstruction_objective.py` |
| 독립 장복원 decoder D_rec | `src/climate_manifold/downstream/observed_decoder.py` |
| 선택적 미래 분위수 변화율 감독 | `src/climate_manifold/downstream/statistical_flow.py` |
| 관측 조건부 Flow Matching·관측만 사용하는 sampler | `src/climate_manifold/downstream/conditional_flow.py` |
| 조건부 분위수 시나리오 NPZ·설정 JSON 출력 | `src/climate_manifold/downstream/sample_conditional_flow.py` |
| 물리 잔차와 선택적 tendency 감독 | `src/climate_manifold/hybrid_pinn.py` |
| 예측 E→F→D 및 별도 장복원 decoder D_rec 구성 | `src/climate_manifold/downstream/pipeline.py` |
| 전체 쌍별 실행 | `scripts/run_pairwise_manifold_comparison.sh` |
| 평가 및 조합별 집계 | `src/climate_manifold/downstream/evaluate.py`, `compare.py` |

## Statistical 선택: W2 또는 KL–entropy

기존 W2는 기본값으로 유지합니다. Statistical이 활성화된 두 조합
(`pinn_statistical`, `statistical_static`)에서만
`--statistical-loss w2|kl_entropy`를 선택합니다.
Raw와 `pinn_static`에는 이 항이 없습니다. 이전 checkpoint의 누락된 선택값은
W2로 해석하며, 기존 `--distribution-weight` 학습 경로는 변경하지 않습니다.

- **W2:** 기존 면적 가중 공간 주변분포의 32개 분위수 제곱 차이.
- **KL–entropy:** 관측 분포 P와 복원 분포 Q에 대해
  `KL(P || Q) = H(P,Q) - H(P)`를 최소화합니다.
  단순히 `|H(P)-H(Q)|`를 줄이거나 복원 entropy를 최대화하는 항이 아닙니다.
  서로 위치가 다른 분포가 같은 entropy를 가질 수 있기 때문입니다.

각 샘플·관측 시각·변수별로 공간 격자값의 분포를 만듭니다. 기존 정규화 좌표에서
고정된 공통 bin 경계와 sigmoid soft membership을 사용하고 격자 면적으로 가중합니다.
기본값은 64 bins(양 끝의 열린 tail bin 포함), 경계 범위 [-6,6], bandwidth 0.2입니다.
각 bin에 1e-6을 더한 뒤 정규화하여 log(0)을 방지합니다. 경계는 예측값이나 검증자료에
맞춰 이동하지 않습니다. 관측 target은 detach하며, 복원값을 통해 E와 보조 decoder에
gradient가 전달됩니다. `--kl-bins`, `--kl-range`, `--kl-bandwidth`는 KL에서만 사용합니다.

현재 Statistical의 변수 범위는 유지됩니다. D_rec가 복원하는 전체 surface 변수
(해면기압 포함)와 D_I의 dynamic information을 사용하며 static 지형은 제외합니다.
두 경로가 활성화되면 두 손실을 1:1 평균합니다.
`information_only`에서는 dynamic information만 사용합니다.
특히 surface는 기존 **격자별 평균·표준편차로 정규화한 값**이므로 물리 단위 Pa의
공간 기압 분포 자체와 같지 않습니다. 이 선택은 msl 전용 손실로 변경하는 기능이 아닙니다.
이산 entropy의 단위는 nats이며, 기상학적 열역학 entropy나 앙상블 예측 불확실성이 아닙니다.

학습 기록에 `statistical_total`, `statistical_kl_entropy`,
`statistical_target_entropy`, `statistical_reconstructed_entropy`,
`statistical_cross_entropy`를 저장합니다. W2 전용
`information_spatial_quantile`에는 KL 값을 넣지 않습니다.
선택과 histogram 설정은 checkpoint·평가 보고서에도 보존합니다.

하나만 실행할 경우 기존 실행 환경에서 다음과 같이 선택합니다.

```bash
STATISTICAL_LOSS=w2 bash scripts/run_pairwise_manifold_comparison.sh
# 또는
STATISTICAL_LOSS=kl_entropy bash scripts/run_pairwise_manifold_comparison.sh
```

두 방법을 같은 seed와 예측기로 비교하려면 새 RUN 경로와 자료 경로를 지정하고:

```bash
MODELS="mlp neural_ode climode convlstm simvp" \
PAIRS=pinn_statistical SEEDS=7 BATCH_SIZE=16 \
STATISTICAL_LOSSES="w2 kl_entropy" \
bash scripts/run_pairwise_manifold_comparison.sh
```

`STATISTICAL_LOSSES`는 단일 `STATISTICAL_LOSS`보다 우선합니다.
각 예측기마다 Raw 1회 + W2 1회 + KL 1회, 총 15회 학습합니다.
세 제약 조합을 모두 비교하면 예측기당 6회입니다.
Raw와 PINN+Static을 loss 종류마다 반복 학습하지 않습니다.
W2 파일명은 유지하고 KL 파일명에 `-kl_entropy`를 추가합니다.
집계에서는 KL arm을 `pinn_statistical:kl_entropy`처럼 구분합니다.
동일 제약 조합의 W2→KL 효과는 `comparison.statistical-effects.csv`에 기록하며,
양수 RMSE/CRPS skill은 KL 후보에 유리합니다.
loss 종류와 제약 조합을 동시에 바꾼 비교는 단일 요인의 효과로 집계하지 않습니다.
같은 loss 종류의 histogram 설정이 다르면 seed 반복으로 합칠 수 없습니다.

W2와 KL의 수치 크기는 직접 비교할 수 없습니다. 먼저 같은 외부 weight로 통제한 비교를
하고, 필요하면 훈련/selection 분할에서만 각 weight를 조정하여 최종 검증 예측력을 비교합니다.
KL의 soft histogram은 근사이며 bin과 bandwidth에 민감합니다. 열린 tail bin은 범위 밖
이상치의 크기 차이를 구분하지 못하고 sigmoid가 포화되면 gradient가 약해질 수 있습니다.
값의 공간 위치도 이 주변분포 손실만으로는 보존되지 않으므로 pointwise reconstruction과
PINN을 함께 평가해야 합니다.

선택 손실 추가 후 전체 suite 542개와 KL 단독 gradient 경로 테스트 1개가 통과했습니다.
합성 자료 학습·checkpoint 재로딩·평가, 이전 W2의 수치 동일성, KL gradient와 entropy 항등식,
비교군 분리 및 Raw 중복 실행 방지를 확인했습니다. 실제 자료 성능 비교는 실행하지 않았습니다.

## 선택적 현재→미래 분포 flow 손실

`--statistical-flow-weight`를 양수로 지정하면 **기존 관측 복원의 W2/KL 손실을 유지하면서**,
예측기가 만든 미래 분포의 변화율에 대한 보조 손실을 추가합니다. 기본값 0은 기존 경로와 같습니다.
`pinn_statistical`과 `statistical_static`의 공동 latent 학습에서만 지원하며,
Raw와 `pinn_static`에는 추가하지 않습니다. W2와 KL 중 어느 Statistical 손실을 선택해도
같은 flow 항을 사용할 수 있으므로 복원 분포의 척도와 시간 변화 감독의 효과를 나누어 비교합니다.

각 샘플·변수·시각에서 격자 면적으로 가중한 정규화 값의 공간 주변분포를 구성합니다.
그 역누적분포의 중간 분위수들을
`Q_k(q_j)`, `q_j=(j+0.5)/J`로 나타냅니다. 기본 `J=32`이며,
`--statistical-flow-quantiles`로 1–512 범위에서 설정합니다. 실제 미래 분포는 `Q_k`, 예측 미래 분포는
`Q̂_k`, 관측 origin은 `Q_0`입니다. 미래 분포는 F의 예측 latent를 **D_rec와 D_I**로
복원해서 얻습니다. 정답 미래를 encoder에 넣어 미래 latent를 만드는 방식이 아닙니다.

\[
\widehat Q_0=Q_0,\qquad
v_k(q_j)=\frac{Q_k(q_j)-Q_{k-1}(q_j)}{\Delta t_k},\qquad
\widehat v_k(q_j)=\frac{\widehat Q_k(q_j)-\widehat Q_{k-1}(q_j)}{\Delta t_k},
\]
\[
L_{\mathrm{flow,field}}
=\operatorname{mean}_{\text{sample},k,\text{variable},j}
\left|\widehat v_k(q_j)-v_k(q_j)\right|^2.
\]

`Δt_k`는 origin부터 시작하는 각 예측 lead 간 실제 시간 간격이며 **일(days)** 단위입니다.
첫 미래 구간은 두 경로 모두 동일한 실제 origin 분포에서 시작합니다. 이후에는 각각의
예측 분포와 실제 분포 사이 변화율을 비교하므로 변화 방향과 크기의 시간적 오차를 감독합니다.
이 정의는 per-step 평균이고 시간 적분의 quadrature는 아닙니다.
해당 변수의 정규화 좌표를 사용하므로 손실의 단위는 정규화 값²/day²이며,
Pa/day 단위의 기압 속도를 직접 비교하는 것은 아닙니다.

기본 분리 모드와 이전 D 공유 모드는
`L_flow = 0.5 × (L_flow,surface + L_flow,dynamic_information)`입니다.
`information_only`에서는 dynamic information 항만 사용합니다. 정적 지형은 제외하고,
surface가 활성화되면 `msl`을 포함한 archive의 모든 변수를 사용합니다.
별도의 신경망이나 새로운 decoder는 만들지 않습니다.

\[
L_{\mathrm{total}}=L_{\mathrm{기존}}+
\lambda_{\mathrm{flow}}L_{\mathrm{flow}}.
\]

여기서 `L_기존`에는 예측·미래 tendency·관측 복원·선택한 PINN/Statistical/Static이 그대로
포함됩니다. **flow weight는 `--statistical-weight`와 독립적**이며 Statistical weight를
다시 곱하지 않습니다. KL을 선택하더라도 flow는 분위수 변화율 손실이고,
두 시각의 KL 값을 빼는 방식이 아닙니다.

```mermaid
flowchart TD
    E["관측 history → E"] --> F["F: 미래 latent"]
    F --> D["예측 D: 미래 기상장"]
    F --> DR["D_rec: 미래 기상장 복원"]
    F --> DI["D_I: 미래 동적 정보 복원"]
    DR --> L["분위수 변화율 손실"]
    DI --> L
    T["실제 origin·미래 분포: 정답만"] --> L
```

flow만 역전파해도 E/F와 활성 보조 decoder에 gradient가 전달됩니다. 기본 분리 모드에서는
예측 D에 직접 전달하지 않고, 이전 `surface_and_information` 모드에서는 D도 갱신합니다.
관측 PINN/복원/Statistical의 입력과 gradient 경로는 바뀌지 않습니다. 미래 상층 정보는
학습 시 정답으로만 사용하고, 추론에는 과거 관측과 origin 정보만 필요합니다.
실제 예측 출력은 여전히 `E → F → D`이며 보조 decoder는 추가 추론 입력을 요구하지 않습니다.

학습 로그에는 `statistical_flow`, `statistical_flow_surface`, `statistical_flow_information`,
그리고 외부 weight를 곱한 `statistical_flow_regularization`을 기록합니다.
checkpoint와 평가 보고서의 `statistical_flow_config`에는 weight·분위수 개수·시간 단위와
origin 기준을 보존합니다. 이전 checkpoint처럼 이 설정이 없으면 꺼진 상태로 해석합니다.

### 실행과 비교

하나의 설정만 실행하려면 기존 자료 환경에서 새 RUN 경로를 지정하고:

```bash
RUN=runs/pinn_statistical_kl_flow \
PAIRS=pinn_statistical SEEDS=7 BATCH_SIZE=16 \
STATISTICAL_LOSS=kl_entropy STATISTICAL_FLOW_WEIGHT=0.1 \
bash scripts/run_pairwise_manifold_comparison.sh
```

W2/KL 각각에 대해 flow on/off를 비교하려면:

```bash
RUN=runs/pinn_statistical_flow_comparison \
MODELS="mlp neural_ode climode convlstm simvp" \
PAIRS=pinn_statistical SEEDS=7 BATCH_SIZE=16 \
STATISTICAL_LOSSES="w2 kl_entropy" STATISTICAL_FLOW_WEIGHTS="0 0.1" \
STATISTICAL_FLOW_QUANTILES=32 \
bash scripts/run_pairwise_manifold_comparison.sh
```

`STATISTICAL_FLOW_WEIGHTS`는 단일 `STATISTICAL_FLOW_WEIGHT`보다 우선합니다.
runner에는 음수가 아닌 유한 소수(예: `0`, `0.1`, `.05`)를 입력합니다.
지수 표기는 runner에서 사용하지 않으며, 수치는 12개 유효숫자로 정리합니다.
같은 값을 중복 입력하면 학습 전에 오류를 냅니다. 분위수 옵션은 양수 flow가 있는 경우에만
지정할 수 있고, off 실험에는 전달하지 않습니다. 활성 실험의 파일명에는 `-flow0.1` 같은
접미사가 붙으며, 꺼진 실험의 파일명은 기존과 같습니다.

위 예시는 **5개 예측기 × (Raw + W2 + W2·flow + KL + KL·flow) = 25회** 학습합니다.
Raw는 모델·seed별 한 번만 실행합니다. 세 제약 조합을 모두 선택하면 예측기당 10회입니다.
PINN+Static은 flow 및 Statistical 종류마다 반복하지 않습니다.
같은 base Statistical 손실에서 flow weight만 바꾼 실험을 우선 비교하고,
같은 flow 설정에서 W2/KL을 비교합니다. 종류와 flow를 동시에 바꾸면 두 효과가 섞입니다.
다른 분위수 개수를 같은 seed 반복으로 합치지 않습니다.
flow 효과는 `comparison.flow-effects.csv`와 JSON의 `statistical_flow_effects`에 기록합니다.

### 해석의 범위

이 항은 1차원 주변분포의 분위수 대응을 활용한 **수송 속도 감독**입니다.
노이즈에서 자료로 가는 벡터장을 학습하는 생성형 conditional flow matching을 구현한 것은
아니며, 확률 예측의 ensemble 분포나 고차원 기상장 전체의 공간 수송을 뜻하지 않습니다.
모든 값의 공간 위치를 섞어도 분포가 같으면 감지하지 못하며, 변수 사이의 결합분포도 보장하지
않습니다. 따라서 태풍 이동이나 시공간 궤적의 정확도는 기존 예측 손실·평가로 확인해야 합니다.
연속적인 물리 방정식을 푸는 새 solver나 미래 PINN 손실도 추가하지 않습니다.

기본 분리 모드에서 flow는 보조 decoder가 읽은 미래 latent의 분포를 감독합니다.
최종 예측 decoder D의 출력 분포가 자동으로 같은 개선을 얻는다는 보장은 없습니다.
또한 flow를 켜면 **미래 동적 정보에 대한 감독도 추가**되므로 예측력 차이를 변화율 식만의
효과로 해석할 수 없습니다. 더 강한 주장을 위해서는 향후 같은 미래 정답을 쓰면서
시점별 분포만 맞추는 endpoint-only 비교군도 필요합니다. 이번 변경에는 그 비교군을 추가하지 않습니다.
flow를 켠 뒤 보조 손실만 감소했는지, 최종 기상장의 RMSE/ACC와 분포 변화 오차도 개선됐는지
함께 확인해야 합니다. `λ_flow=0.1`은 예시이며 간격²으로 나누는 손실의 크기를 고려해
훈련/selection 구간에서 조정합니다. 검증·시험 자료로 weight를 맞추지 않습니다.

전체 회귀 테스트 601개와 이후 추가한 시간 정합성 4개·runner 분위수 상한 1개 검증이 통과했습니다.
W2/KL 각각의 Neural ODE·ClimODE 합성자료 학습, flow 단독 gradient 경로,
checkpoint 재로딩·평가와 미래 정답의 추론 입력 누출 방지를 확인했습니다.
실제 ERA5 자료의 예측력 향상은 아직 검증하지 않았습니다.

## 관측 쌍 조건부 Flow Matching

`--conditional-flow-weight`를 양수로 지정하면 **관측 origin−6h, origin 쌍**에서 출발하는
conditional flow matching(CFM)을 보조 학습합니다. 기존 예측기 F가 만든 미래 latent를
사용하지 않습니다. 관측 복원의 W2/KL·PINN·Static과 주 예측 손실은 그대로 유지합니다.
CFM은 `pinn_statistical` 또는 `statistical_static`의 joint latent 실험에서 사용할 수 있습니다.
decoder는 `separate_surface_and_information` 또는 `information_only`여야 합니다.
예측 D를 공유하는 `surface_and_information`은 CFM에서 지원하지 않습니다.

| 구분 | `STATISTICAL_FLOW_WEIGHT` | `CONDITIONAL_FLOW_WEIGHT` |
|---|---|---|
| 출발 경로 | F의 예측 미래 latent → 보조 decoder | 관측 쌍 → E와 보조 decoder → 조건부 head |
| 미래 감독 | 예측·실제 분위수의 물리 시간 변화율 | 관측 분포+noise → 실제 미래 분포의 벡터장 |
| 새 네트워크 | 없음 | 보조 CFM head |
| 단독 역전파 | E·F·활성 보조 decoder | E·활성 보조 decoder·CFM head |
| 기본값 | 0 | 0 |

두 weight를 동시에 양수로 주면 오류를 냅니다. 같은 sweep에 양수인 두 Flow 종류를 넣는 것도
금지하여 실험 효과가 섞이는 것을 방지합니다. 기존 예측 Flow 옵션·파일명은 보존합니다.

### 손실과 관측 조건

면적 가중 정규화 주변분포의 분위수를 `Q_t`, 미래 lead `h`의 실제 분위수를 `Q_(t+h)`라 합니다.
CFM 조건 `c_t`는 관측 쌍의 encoder 표현과 **보조 decoder가 복원한 현재 분포**를 사용합니다.
관측 분포의 실제 분위수를 공통 기준점으로 쓰고 noise를 더합니다. 실제 미래값은 flow 학습의
도착점과 정답 벡터장을 만드는 데만 사용하며 관측 조건에는 들어가지 않습니다.

\[
s_0(h)=Q_t+\sigma\epsilon_h,\qquad s_1(h)=Q_{t+h},\qquad
\epsilon\sim\mathcal N(0,I),\quad \tau\sim U(0,1),
\]
\[
s_\tau=(1-\tau)s_0+\tau s_1,\qquad
L_{\rm CFM}=\mathbb E_{\epsilon,\tau}
\left[\|v_\theta(s_\tau,\tau\mid c_t,h)-(s_1-s_0)\|^2\right].
\]

`τ`는 0–1 사이의 생성 Flow 시간이며, 물리 lead `h`는 **일(days)**로 별도 입력합니다.
CFM target은 `s1-s0`이며 물리 lead로 나눈 속도가 아닙니다. 기본 32개 중간 분위수를 사용합니다.
샘플 하나의 전체 lead 구간에 같은 `τ`를 적용하고, 초기 Gaussian 성분은 lead·변수·분위수마다
독립적으로 뽑습니다. head는 lead별 공유 MLP와 kernel 3의 시간축 residual convolution으로
전체 미래 분포 시퀀스의 벡터장을 함께 계산합니다. 시간축 결합은 표현 능력이며 물리적 일관성 보장은 아닙니다.
CFM은 활성 surface와 dynamic information의 분위수를 이어 붙이고 샘플·lead·변수·분위수
전체의 제곱오차를 평균합니다. 기존 Statistical의 두 그룹 1:1 평균과는 달리, CFM의 그룹별
실효 비중은 변수 수에 비례합니다. `information_only`에서는 동적 정보 항만 사용합니다.
고정 지형은 분포 생성 대상에서 제외합니다.
W2와 KL 모두 같은 CFM을 사용할 수 있으며 KL histogram을 직접 이동시키는 방식은 아닙니다.

\[
L_{\rm total}=L_{\rm 기존}+\lambda_{\rm CFM}L_{\rm CFM}.
\]

외부 CFM weight는 Statistical weight와 독립적입니다. CFM만 역전파하면 공유 E, D_rec/D_I 및
CFM head가 갱신됩니다. **F·예측 D·PINN closure는 CFM으로 직접 갱신되지 않습니다.**
기존 예측 손실은 계속 E·F·D를 갱신하므로 전체 학습은 동시 학습입니다.
CFM head의 파라미터는 보조 파라미터 수에 기록하고, 주 예측기의 파라미터 수와 구분합니다.

### 선택과 비교 실행

| 환경 변수 | 기본값 | 허용 범위·의미 |
|---|---|---|
| `CONDITIONAL_FLOW_WEIGHT` | `0` | 단일 CFM 가중치, 0이면 비활성 |
| `CONDITIONAL_FLOW_WEIGHTS` | 단일값 사용 | 예: `"0 0.1"`; 단일 변수보다 우선 |
| `CONDITIONAL_FLOW_QUANTILES` | `32` | 정수 1–512, 변수별 분위수 개수 |
| `CONDITIONAL_FLOW_HIDDEN_DIM` | `128` | 정수 1–4096, 보조 head 폭 |
| `CONDITIONAL_FLOW_NOISE_SCALE` | `0.2` | 유한 양수, 정규화 단위의 초기 noise 표준편차 |

직접 Python 학습에서는 각각 `--conditional-flow-weight`, `--conditional-flow-quantiles`,
`--conditional-flow-hidden-dim`, `--conditional-flow-noise-scale`를 사용합니다.
비활성 상태에서는 head를 생성하지 않고 head 설정 옵션도 받지 않습니다. 이전 checkpoint에
`conditional_flow_config`가 없으면 비활성으로 해석합니다. 선택 설정과 head 구조는 checkpoint와
평가 보고서에 보존합니다. 서로 다른 head 구조의 결과를 같은 seed 반복으로 섞지 않습니다.

기존 ARCHIVE/INFO 환경에서 W2/KL 각각 CFM on/off를 비교하려면:

```bash
RUN=runs/pinn_observed_conditional_flow \
MODELS="mlp neural_ode climode convlstm simvp" \
PAIRS=pinn_statistical SEEDS=7 BATCH_SIZE=16 \
STATISTICAL_LOSSES="w2 kl_entropy" STATISTICAL_FLOW_WEIGHTS=0 \
CONDITIONAL_FLOW_WEIGHTS="0 0.1" \
CONDITIONAL_FLOW_QUANTILES=32 CONDITIONAL_FLOW_HIDDEN_DIM=128 \
CONDITIONAL_FLOW_NOISE_SCALE=0.2 \
bash scripts/run_pairwise_manifold_comparison.sh
```

예측기 5개마다 Raw·W2·W2+CFM·KL·KL+CFM을 학습하므로 총 25회입니다.
Raw와 PINN+Static은 CFM 설정마다 반복하지 않습니다. 활성 파일에는 `-cfm0.1`이 붙으며,
집계 arm은 `pinn_statistical:cfm=0.1`, `pinn_statistical:kl_entropy:cfm=0.1`처럼 구분합니다.
가중치는 runner에서 음수가 아닌 유한 소수로 입력하고 12개 유효숫자로 정리합니다.
`0.1`은 성능을 검증한 권장값이 아니라 비교용 예시입니다.
같은 W2/KL 설정에서 CFM weight만 바꾼 결과는 `comparison.conditional-flow-effects.csv`와
JSON의 `conditional_flow_effects`에 기록합니다. CFM 설정이 같을 때의 W2→KL 비교는 기존
`comparison.statistical-effects.csv`에 남깁니다. CFM과 기존 예측 Flow를 교체한 비교를
한 가지 weight 변화의 효과로 합치지 않습니다.

### 미래 정답 없는 분포 sampling

학습된 CFM head는 관측 쌍과 현재 정보만으로 초기 noise를 바꾸어 여러 시나리오를 만듭니다.
`τ=0 → 1`의 벡터장을 midpoint 방법(기본 32 steps)으로 적분하고 마지막 분위수 축을 정렬하여
각 marginal의 분위수를 단조롭게 만듭니다.
이는 SDE solver가 아니라 **확률적 초기값을 사용하는 ODE sampling**입니다.

```bash
PYTHONPATH=src python -m climate_manifold.downstream.sample_conditional_flow \
  --checkpoint runs/pinn_observed_conditional_flow/neural_ode-pinn_statistical-cfm0.1-seed7.pt \
  --archive "$ARCHIVE" --information "$INFO" \
  --split validation --max-cases 16 --members 8 --steps 32 --seed 7 \
  --output runs/pinn_observed_conditional_flow/neural_ode-cfm-samples.npz
```

새 `.npz`와 같은 이름의 `.json` 보고서를 저장합니다. `normalized_quantiles`의 축은
`[case, member, lead, variable, quantile]`입니다. 변수명·분위수 수준·lead 시간·origin·seed와
정규화 정보를 함께 보존합니다. sampler는 예측기 F와 미래 정답을 사용하지 않습니다.
학습 때 endpoint에 미래 정답이 들어가는 것은 감독 학습이며, inference에는 전달하지 않습니다.

### 결과를 해석할 때

CFM은 미래 분포를 만드는 보조 생성 목적입니다. 기상장 예측력이 좋아진다면 공유 표현을 통한
효과일 수 있지만, **미래 상층 정보 감독과 보조 파라미터도 추가**되므로 개선을 Flow 방식만의
효과로 단정할 수 없습니다. 같은 미래 정답을 쓰는 endpoint 감독·동일 head 용량 비교가
추가로 필요합니다. 기존 최종 기상장 RMSE/ACC와 함께 CFM 생성 분포의 보정·다양성을 별도 평가해야 합니다.

출력은 **정규화된 공간 주변분포**이며 위치가 있는 기상장·태풍 track ensemble이 아닙니다.
격자별로 정규화한 surface의 marginal을 하나의 scalar inverse transform으로 물리 Pa 분포로
바꿀 수 없습니다. 마지막 정렬은 분위수 단조성만 보장하고 확률 보정이나 시간적 물리 일관성을
보장하지 않습니다. 실제 ERA5에서의 정확도·장기 안정성·불확실성 보정은 아직 검증하지 않았습니다.

구현 검증: 전체 회귀 테스트 702개와 이후 추가한 checkpoint decoder 계약 검증 1개가 통과했습니다.
W2/KL × Neural ODE/ClimODE 합성 학습, CFM 단독 gradient 분리, 미래 입력 누출 방지,
기존 파라미터 초기값·비활성 checkpoint 호환, seed 재현성과 분포 sampling을 확인했습니다.
이는 코드 동작 검증이며 실제 자료 예측력·앙상블 보정의 성능 근거는 아닙니다.
