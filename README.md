# Climate Manifold

기후장 표현과 예측기를 **함께 학습**하는 저장소입니다. 독립 A-step 학습도 보존합니다. `climate_diffusion`의
`feature/a64-b512-expanded`에서 A에 필요한 부분을 분리했습니다.
Hydra의 B/C 학습, MoE 전문가, 게이트, 라우터, 전문가 간 결합은 포함하지 않습니다.

| 구성 | 포함 내용 |
|---|---|
| 표현 | DCT + 전역 autoencoder, **manifold 64차원 / 은닉층 폭 512** |
| 물리 동역학 | latent drift의 **다단계 자유 rollout**, 미래 latent·기후장·정보 감독, tendency/AE delta·물리 제약 |
| 정보 | Z850/Z500/Z250, U850/V850, 고정 지형 고도·경사, 정보 복원·분포 손실 |
| Hybrid PINN | 선택적으로 기압면 운동량·열역학·연속·층후 제약 및 learned closure |
| A 보조 sampler | raw latent에서 flow matching, 상태·전이 분포 및 120시간 경로 손실 |
| 데이터/평가 | ERA5 변환·shard 다운로드, train-only 정규화, 지연시간별 진단·CRPS·geometry audit |
| 예측 주실험 | **Encoder → latent MLP/Neural ODE → decoder 공동 학습**; 같은 구조에서 물리·정보 제약 off/on 비교 |

**512는 manifold 차원이 아니라 MLP 은닉층 폭입니다.** 잠재 상태는 격자별 64채널이
아닌 전체 입력 기후장을 압축한 단일 64차원 벡터입니다. A 보조 sampler는 원래 A의
분포 학습에 필요하므로 유지했으며, Hydra의 B 모델과는 별개입니다.

## 설치 및 합성 검증

Python 3.10 이상, Linux/macOS 환경을 권장합니다. CUDA 학습은 사용 환경에 맞는
PyTorch를 먼저 설치하세요. 아래 명령은 저장소 루트에서 실행합니다.

```bash
git clone https://github.com/nayehyeon61-glitch/climate_manifold.git
cd climate_manifold
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,era5]'
python -m pytest -q
python scripts/smoke_climate_manifold.py --output runs/smoke-a64
```

Smoke는 실제 기본 크기 **64/512**에서 PINN warm-up 1 epoch + A curriculum 6 epochs,
다단계 drift 학습, checkpoint 재로딩, 120시간 보조/기존 anchored drift/새 pure drift 예측,
tangent/AE audit를 확인합니다.
빠른 소형 확인에는 `--tiny`를 추가합니다. 출력 디렉터리는 매번 새 경로를 사용합니다.
합성 필드는 소프트웨어 검증용이며 ERA5 예측력의 근거가 아닙니다.

## 독립 A 학습 — 선택적 실험

기존 6시간 `surface.npz` + 같은 이름의 `.schema.json`, 그리고 정합된 정보
`.npz` + sidecars 또는 compact-shard 디렉터리를 그대로 사용할 수 있습니다.
PINN 활성화에는 같은 기압면의 U/V/T/Z/omega와 실제 표면기압 `sp`가 필요합니다.
기존 Z/지형 정보만 있는 파일은 PINN 입력으로 충분하지 않습니다.

```bash
export ARCHIVE=/absolute/path/to/surface.npz
export INFO=/absolute/path/to/pinn_information_shards
export RUN=runs/manifold_a64_pinn_001
export PINN=1 DEVICE=cuda
bash scripts/run_climate_manifold.sh preflight
bash scripts/run_climate_manifold.sh train
bash scripts/run_climate_manifold.sh audit
bash scripts/run_climate_manifold.sh validation
bash scripts/run_climate_manifold.sh pure-drift-validation
```

이 명령은 **A만** 학습합니다. 아래 공동 예측 학습의 필수 선행 단계가 아닙니다. 기본은 60 epochs, batch 2, members 4, tau steps 4,
manifold 64, hidden 512, 6시간 × 20 전이입니다. PINN은 1 epoch closure warm-up 후
3 epochs에 걸쳐 가중치를 올립니다. A의 6단계 curriculum 간격은 4 epochs이며,
전체 활성화 후에만 최적 checkpoint를 선택합니다. `PINN=0`이면 PINN 없이 정보·동역학
학습을 수행합니다. 정보가 없는 surface 실험은 `MODE=surface PINN=0`을 사용합니다.

`process`·`dynamics` profile에는 **A 자체의 미래 표현 학습**이 기본 적용됩니다.
현재 raw latent에서 기존 drift를 반복 적용하고, 같은 origin 정보로 인코딩한 미래 latent와
순수 decoder의 미래 기후장·정보를 감독합니다. 새 큰 예측기를 추가하지 않고
encoder·decoder·drift를 함께 학습합니다. 기본 직접 감독 구간은 **6 → 12 → 24시간**이며
기존 120시간 ensemble 학습도 유지합니다. `DYNAMICS_MAX_STEPS=0`은 새 경로를 끄고
기존 A 손실 동작을 재현하는 ablation입니다. [구조·손실·학습/평가 계약](docs/dynamics_training.md)을 참고하세요.

직접 명령도 가능합니다.

```bash
train-climate-manifold --archive "$ARCHIVE" --information "$INFO" \
  --output "$RUN/manifold.pt" --manifold-dim 64 --hidden-dim 512 \
  --epochs 60 --pinn --device cuda
```

`--profile baseline|dynamics|information|process`로 기존 A ablation 구성을 선택합니다.
전체 구성은 `process`이며, 다른 profile의 sampler는 분포 손실로 학습되지 않습니다.
그 경우 확률예측 비교에는 사용하지 말고 drift/재구성 평가를 우선하세요.

## 데이터 준비

지상 입력은 순서까지 고정된 `msl,t2m,u10,v10`의 완전 관측 6시간 장입니다.
같은 격자·정확한 시간·단위를 검사하고 누락값을 임의로 채우지 않습니다.

```bash
prepare-climate-manifold-data --fields /path/to/surface.nc \
  --variables msl t2m u10 v10 --step-hours 6 \
  --target-lat-points 18 --target-lon-points 36 --output data/surface.npz

# 요청 계획만 확인; 이 명령은 다운로드하지 않습니다.
python scripts/stream_era5_extra.py --archive data/surface.npz \
  --store data/information_pinn --pinn

# CDS 인증/데이터셋 접근 설정 후 실제 다운로드·변환
python scripts/stream_era5_extra.py --archive data/surface.npz \
  --store data/information_pinn --pinn --download --delete-raw
```

`--delete-raw`는 변환·checksum 확인이 끝난 해당 store 소유의 원본만 지웁니다.
중단된 다운로드는 같은 store로 재개할 수 있습니다. `PINN=1` 데이터는 기존
non-PINN store와 다른 디렉터리를 사용합니다. 이미 모든 추가 필드가 NetCDF로
있다면 `prepare-climate-manifold-information --archive ... --fields ... --output ... --pinn`
으로 sidecar를 만들 수 있습니다.

다운로드와 A 학습을 겹치려면 새 `RUN`을 정한 후 실행하세요.

```bash
export ARCHIVE=/absolute/path/to/surface.npz
export INFO=/absolute/path/to/new_pinn_shards
export RUN=runs/manifold_stream_001
export PINN=1 DEVICE=cuda
bash scripts/run_streaming_a_information.sh
```

A 학습·선택에 필요한 shard가 준비되면 학습을 시작합니다. 종료 전 남은 데이터
발행까지 기다립니다. B/C는 실행하지 않습니다. 학습 중단 시 optimizer/RNG 복구는
지원하지 않으므로 데이터 store를 재사용하되 새 `RUN`에서 학습합니다.

## 결과 해석 및 재사용

- `manifold.pt`: 최적 A 가중치, 정규화·정보 통계·분할·설정. `.manifest.json`과 함께 보관합니다.
- `manifold.metrics.json`: epoch별 원손실, 가중손실, 고정 validation selection.
- `geometry-audit.json`: AE delta, 학습 drift, tangent oracle의 실제 단위 시간 변화 진단.
  AE/tangent oracle은 관측된 미래를 쓰므로 예측 성능이 아닙니다.
- `validation.json`: CRPS, transition CRPS, RMSE, spread, coverage, persistence 비교.
- `validation.npz`: 첫 평가 초기시각의 물리 단위 `[members,20,state_dim]` 예측 및 실제 valid time.
- `pure-drift-validation.json/.npz`: 초기장 복원 잔차를 더하지 않은 **encoder → A drift → decoder** 평가.

`run_climate_manifold.sh drift-validation`은 residual을 정확히 0으로 둔 drift 대조 실험입니다.
점수 인터페이스를 위해 같은 경로를 복제할 뿐 확률 앙상블은 아닙니다.
이 기존 명령에는 여전히 초기장 복원 잔차가 더해집니다. 새 명령인
`pure-drift-validation`은 이를 제거한 decoder 출력으로 평가하며, 첫 6시간 오차에는
초기 재구성 오차의 영향도 포함됩니다. 두 평가 경로를 같은 방식의 결과로 혼동하지 않습니다.
설정 확정 후에만 `run_climate_manifold.sh test`로 최종 test를 평가합니다.

원본 결과와 분할을 맞추기 위해 `train / expert_validation / calibration / validation / test`
이름과 5-way embargo를 유지합니다. 여기서 `expert_validation`은 **A checkpoint 선택용**
이름이며 전문가 모델을 뜻하지 않습니다. `calibration` 구간은 독립 A 학습에서 사용하지
않으며 후단 예측기 실험에서는 checkpoint 선택에 사용합니다.
정규화는 train에서만 적합하고 평가 시 checkpoint에 고정된 통계를 씁니다.
미래 정보는 손실의 label로만 쓰며, rollout 조건에는 시작시각 정보만 들어갑니다.

독립 A의 기존 보조/anchored drift 예측은 `decode(z_t) + x_origin - decode(z_origin)`로 원점을 고정합니다. 따라서
출력은 decoder manifold의 원점별 평행이동 위에 있으며, 모든 예측을 하나의 동일한
decoder image로 엄밀히 투영했다고 해석해서는 안 됩니다.
반면 새 A 다단계 직접 감독·pure drift 평가와 후단 주실험은 **decoder 출력만으로 예측**합니다.
기존 A checkpoint는 그대로 불러올 수 있지만, 새 동역학 감독을 학습한 것으로 바뀌지는 않습니다.

표현을 다른 NN/ODE에 연결할 때:

```python
from climate_manifold.train import load_checkpoint
model, metadata = load_checkpoint("runs/example/manifold.pt")
model.eval()
# x와 information은 checkpoint 통계로 정규화된 tensor입니다.
z = model.raw_encode(x, information)  # [batch,64]
x_reconstructed = model.core.manifold.decode(z)
q = model.encode(x, information)      # sealed train mean/scale로 표준화된 좌표
```

## Manifold와 예측기 공동 학습

기본 실험은 **encoder → latent 예측기 → decoder 전체를 한 번에 학습**합니다.
미래 기상장 손실이 세 구성요소 모두로 역전파됩니다. A 사전학습, frozen A,
Plain AE 사전학습은 필요하지 않습니다. 현재 global latent 64 / hidden 512 구조를
그대로 사용하며 이번 변경이 공간 격자·메시 encoder를 새로 구현한 것은 아닙니다.

```bash
export ARCHIVE=/absolute/path/to/surface.npz
export INFO=/absolute/path/to/information_pinn_shards
export RUN=runs/joint_comparison_001
export DEVICE=cuda EPOCHS=20 SEEDS='7 19 43'
export PINN=1
bash scripts/run_model_comparison.sh
```

PINN에 필요한 전체 필드가 없으면 `PINN=0`으로 실행합니다. Surface-only는 `INFO`를
설정하지 않습니다. 기본 **2개 예측기 × 2개 손실 구성 × 3개 seed**를 처음부터 학습합니다.
각 pair는 같은 encoder·예측기·decoder 초기화, 입력, 차원, 예측/변화량/재구성 손실을
공유하며 추가 물리·정보 제약만 off/on합니다. `MODELS=neural_ode`로 범위를 줄일 수 있습니다.
각 seed에서 표현도 다시 학습합니다. `A_CHECKPOINT`는 선택 사항이며, 가중치를
재사용하려면 `INITIALIZATION=pretrained`를 명시합니다. 그 경우에도 joint에서는
encoder·decoder가 고정되지 않습니다.

[공동 학습 구조·손실·단일 실행 명령](docs/joint_training.md)과
[실험 계약·평가·기존 frozen 대조군](docs/downstream.md)을 참고하세요.

ClimODE는 격자 미분이 필요하므로 **E → D → ClimODE** 공동 학습을 별도 보조 실험으로
지원합니다. 이것은 latent 안에서 ClimODE가 동작하는 구조가 아닙니다.
`scripts/run_climode_comparison.sh`는 raw ClimODE와 이 경로를 비교합니다.

**평가 기준은 ClimODE 방식의 변수·lead별 RMSE/ACC, 확률 출력의 CRPS입니다.**
`bash scripts/run_climode_benchmark.sh`는 같은 데이터의 Raw ClimODE 기준선과
공동 예측 실험을 연결하고 개선율 CSV를 만듭니다. 정렬된 실제 지형·육해 마스크
`CONSTANTS`가 필요합니다. [평가 정의와 원논문과의 차이](docs/climode_evaluation.md)를 참고하세요.
기존 checkpoint를 재평가할 수 있지만, 재평가만으로 공동 학습된 모델이 되지는 않습니다.

PINN은 희소 기압면의 근사 물리 제약이며 완전한 primitive-equation solver가
아닙니다. 실제 장기 안정성·태풍 이동·앙상블 보정 성능은 별도 실험이 필요합니다.
기존 Hydra checkpoint는 새 모델로 자동 재해석하지 않습니다.

원본 commit, 파일 대응, 추출 검증은 [SOURCE.md](SOURCE.md)를 참고하세요.
PINN의 방정식·단위·mask·gradient 범위는 [물리 설명](docs/physics.md)에 정리했습니다.
