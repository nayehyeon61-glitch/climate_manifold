# 공식 기상모델 구조를 사용하는 비교군

브랜치: `feature/official-weather-baselines`

## 이번에 추가한 실행 경로

| 선택 이름 | 사용 코드 | Raw → 예측 | E → 모델 → D 공동학습 |
| --- | --- | --- | --- |
| `fourcastnet` | NVIDIA 공식 FourCastNet AFNO backbone + 관측 문맥 어댑터 | 지원 | 지원 |
| `climax` | Microsoft 공식 ClimaX 변수 tokenization·aggregation·ViT backbone | 지원 | 지원 |
| `graphcast_official` | 공식 GraphCast JAX/Haiku 모델, 별도 실행 환경 | 별도 runner | 미지원 |

FourCastNet·ClimaX는 원본 핵심 네트워크 코드를 저장소에 포함했습니다. 원본 commit,
라이선스와 수정 범위는 각각 [_vendor/fourcastnet/UPSTREAM.md](../src/climate_manifold/_vendor/fourcastnet/UPSTREAM.md),
[_vendor/climax/README.md](../src/climate_manifold/_vendor/climax/README.md)에 있습니다.
**공식 사전학습 가중치를 불러오는 방식이 아닙니다.** 현재 격자와 변수, 작은 width/depth,
동일한 학습 데이터로 새로 학습하는 구조 비교입니다. 원논문의 성능을 재현했다고 보고하면 안 됩니다.

기존 `climode`의 `raw_backend=matched` 및 latent 경로는 우리가 수정한 transport 모델입니다.
공식 ClimODE의 vendored 모듈을 사용하는 legacy raw/decoded 경로와 구분해야 합니다.

## 입력과 시간 처리

- FourCastNet: AFNO가 현재 상태, 고정된 관측 history, 실제 관측 간격, 달력,
  목표 lead와 origin information을 받아 6시간마다 자기회귀 예측합니다.
  history가 24시간 간격이라고 해서 이를 6시간 간격으로 재해석하지 않습니다.
  건너뛴 lead를 요청해도 중간 6시간 상태는 계산합니다.
- ClimaX: 공식 구조에 따라 **마지막 관측장**과 lead embedding(`lead_hours / 100`)으로
  각 미래 시각을 직접 예측합니다. 앞선 history나 calendar를 사용하는 모델이 아닙니다.
  Raw의 추가 origin information은 변수 token으로 입력됩니다.
  Latent 채널은 물리 변수 자체가 아닌 E가 학습하는 공간 표현입니다.
- Raw와 latent 모두 학습 구간에서 정한 정규화를 사용하며 미래 information은 예측기 입력에
  전달하지 않습니다. Latent의 origin information은 E를 통해서만 들어갑니다.
- 공간 격자는 `WEATHER_PATCH_SIZE`로 나누어져야 합니다. 기본 patch=2는
  Raw 16×32와 latent 8×16에 맞습니다. 다른 격자는 patch=1 등을 명시하세요.
  ClimaX `HIDDEN_DIM`은 4의 배수여야 합니다.
- FourCastNet은 원본 2D FFT의 양 축 주기성을 유지합니다. 이것을 구면 연산자나
  비주기 위도 경계로 해석하지 않습니다. ClimaX는 원본 절대 위치 embedding을 사용합니다.

## 설치와 실행

기존 clone에서 다음과 같이 받습니다. 변경 중인 로컬 파일이 있으면 먼저 보존하세요.

```bash
git fetch origin
git switch --track origin/feature/official-weather-baselines
python -m pip install -e '.[test,era5,forecast]'
```

이미 해당 local branch가 있으면 `git switch feature/official-weather-baselines` 후
`git pull --ff-only`를 사용합니다. PyTorch용 두 모델에는 timm/torchvision 추가 설치가 필요 없습니다.

기존 archive와 정렬된 information 경로를 지정한 뒤, **새 모델 두 개**의
PINN+Statistical(W2) / Raw 비교를 실행합니다. `RUN`은 존재하지 않는 새 경로여야 합니다.

```bash
export ARCHIVE=/absolute/path/to/surface_archive.npz
export INFO=/absolute/path/to/information_archive
export RUN=runs/weather_baselines_w2_$(date +%Y%m%d_%H%M%S)
MODELS="fourcastnet climax" \
PAIRS=pinn_statistical STATISTICAL_LOSSES=w2 \
STATISTICAL_FLOW_WEIGHTS=0 CONDITIONAL_FLOW_WEIGHTS=0 \
INCLUDE_RAW=1 BATCH_SIZE=16 SEEDS="7 19 43" \
EPOCHS=20 HIDDEN_DIM=128 WEATHER_DEPTH=4 WEATHER_PATCH_SIZE=2 \
HISTORY_STEPS=6 HISTORY_STRIDE=4 HORIZON_STEPS=20 \
WINDOW_STRIDE=4 ORIGIN_STRIDE=1 DEVICE=cuda \
bash scripts/run_pairwise_manifold_comparison.sh
```

이 명령은 **2모델 × 2경로 × 3seed = 12회** 학습합니다.
원하는 GPU/CPU에 맞게 `DEVICE`를 지정하세요. 이전 실험이 `WINDOW_STRIDE=1`이었다면
동일 조건으로 재비교할 때도 1을 사용해야 합니다.

7개 모두를 실행하려면 `MODELS="mlp neural_ode climode convlstm simvp fourcastnet climax"`로
변경합니다. W2만 선택하면 42회, `STATISTICAL_LOSSES="w2 kl_entropy"`이면
**63회**입니다(모델·seed마다 Raw는 한 번만 학습). 기본 목록은 기존 2종으로 유지했습니다.
Flow는 기존 결정대로 기본 OFF이며, W2/KL 선택과 별개입니다.

각 실행은 calibration 구간으로 checkpoint를 선택하고 validation 예측과
ClimODE 공통 평가 지표를 생성합니다. 테스트 구간은 다음 명령으로 별도 평가합니다.

```bash
python -m climate_manifold.downstream.evaluate \
  --checkpoint "$RUN/fourcastnet-raw-seed7.pt" \
  --archive "$ARCHIVE" --information "$INFO" --split test \
  --output "$RUN/fourcastnet-raw-seed7.test.json" \
  --forecast-output "$RUN/fourcastnet-raw-seed7.test.npz" --device cuda
```

GraphCast는 [공식 JAX 실행 안내](GRAPHCAST_OFFICIAL.md)의 별도 설치·학습 명령을 사용합니다.
`MODELS=graphcast`를 PyTorch pairwise runner에 넣으면 해당 안내를 출력하고 중단합니다.

## 해석과 검증

학습 checkpoint 및 평가 JSON의 `predictor_provenance`에 공식 repository/commit,
수정 구현, depth, patch size, hidden width, pretrained 여부를 기록합니다.
같은 family에 다른 버전·설정을 섞으면 비교기가 거절합니다.

Raw와 E→모델→D는 채널·공간 해상도와 총 파라미터 수가 다릅니다. 이 비교는
**Manifold를 포함한 전체 시스템의 효과**이며 동일 용량 비교나 PINN 단독 효과가 아닙니다.
현재 pairwise runner에는 제약 없는 E→모델→D 대조군이 없습니다.
기존 `run_model_comparison.sh`에도 새 두 모델을 등록했지만, 그 runner는 이전의
forecast-trajectory 제약 실험이므로 새 observed two-route 결과와 섞지 마세요.

모든 새 모델은 현재 deterministic 예측입니다. 앙상블이나 확률 보정을 구현한 것으로
표시하지 않으며, Gaussian CRPS를 임의로 생성하지 않습니다.

작은 합성 데이터 검증은 실행·gradient·입출력의 확인이며 예측 성능 근거가 아닙니다.
실제 ERA5 전체 학습 및 GPU 메모리 한도는 사용 환경에서 확인해야 합니다.
Pangu-Weather·NowcastNet을 현재 데이터에 자동 추가하지 않은 이유는
[외부 모델의 데이터 요구사항](EXTERNAL_WEATHER_MODELS.md)에 있습니다.
