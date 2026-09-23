# Manifold와 예측기의 공동 학습

실험의 단위는 **encoder E + 실제 사용할 예측기 F + decoder D**입니다.
A를 먼저 학습하고 고정해야 한다는 조건을 제거했습니다. 기본 `--training-mode joint
--initialization fresh`에서는 세 구성요소를 새로 초기화하고 같은 미래 예측 목표로 함께 학습합니다.

```mermaid
flowchart TD
  H["관측 history + origin 정보"] --> E["학습하는 encoder E"]
  E --> Z["과거 latent sequence"]
  Z --> F["학습하는 MLP / Neural ODE F"]
  F --> P["미래 latent sequence"]
  P --> D["학습하는 decoder D"]
  Z --> D
  D --> L["미래 예측 + 현재 복원 손실"]
  P --> I["정보 decoder / 물리 제약"]
  I --> L
```

미래 예측 오차는 D → F → E로 역전파됩니다. 정보 encoder, 정보 decoder 및 PINN
closure도 해당 손실이 활성화되면 함께 학습합니다. 독립 A 학습에 있던 보조 flow sampler와
A 자체 drift를 F 앞에 다시 연결하지 않습니다. 이 실험에서 시간 전개를 담당하는 것은 F입니다.

현재 E/D는 **DCT + 전역 MLP**입니다. 기본 latent 64는 한 기상장 전체를 나타내는
벡터 차원이고, hidden 512는 은닉층 폭입니다. 공간 latent/mesh 표현으로 바뀐 것은
아니므로 공동 학습 이후에도 표현 용량은 별도로 검증해야 합니다.

## 공통 예측 목표와 선택적 제약

\[
L=L_{\mathrm{forecast}}+\lambda_\Delta L_{\mathrm{tendency}}
 +\lambda_{\mathrm{rec}}L_{\mathrm{reconstruction}}
 +\lambda_{\mathrm{phys}}L_{\mathrm{surface\ physics}}
 +\lambda_I L_{\mathrm{information}}
 +\lambda_S L_{\mathrm{static}}
 +\lambda_Q L_{\mathrm{spatial\ distribution}}
 +\lambda_P L_{\mathrm{PINN}}.
\]

| 항목 | 기본 가중치 | 역할 |
|---|---:|---|
| Future forecast | 1 | 자유 rollout으로 예측한 미래 기상장과 실제 미래 비교 |
| Tendency | 0.1 | origin부터 각 인접 lead까지의 변화량 감독 |
| Reconstruction | 0.1 | 관측 입력의 표현 보존; off/on 대조군 모두 유지 |
| Surface physics | 0.01 | 실제 미래와 예측의 발산·와도·기압/기온 기울기·운동에너지 진단 비교 |
| Information | 0.1 | 정보 head의 복원 및 예측 latent에 대한 정보 감독 |
| Static | 0.05 | 예측 latent에서 고정 지형 정보 보존 |
| Spatial distribution | 0 | 선택적 결정론적 공간 분포 매칭; 앙상블 CRPS와 다름 |
| Hybrid PINN | 설정에 따름 | PINN이 활성화된 경우 저장/생성한 PINN 설정의 가중치 사용; `--pinn-weight`로 조정 |

Surface physics는 train 자료로 정한 scale을 사용합니다. 이는 실제 장의 물리적
진단량을 맞추는 감독이며 정확한 PDE 보존을 증명하지 않습니다. Hybrid PINN은 별도
기압면 방정식 잔차/closure 항이며 필요한 U/V/T/Z/omega/sp 자료가 있어야 합니다.
기존 독립 A의 flow matching·ensemble state/transition CRPS를 이 결정론적 학습 경로에
자동으로 이식하지 않습니다. `--distribution-weight`는 그 CRPS를 켜는 옵션이 아닙니다.

`--regularization none`은 추가 surface physics·information·static·distribution·PINN을
끄고 **동일한 future/tendency/reconstruction 목표를 유지**합니다. `full`은 설정된
가중치를 사용합니다. 따라서 공정한 최소 실험은 같은 E–F–D에서 `none` vs `full`입니다.
새로운 predictor마다 표현도 그 predictor와 공동 학습합니다.

## A 사전학습 없이 단일 모델 실행

저장소 루트에서 실행합니다. `surface.npz`와 schema sidecar가 필요합니다.

```bash
python -m climate_manifold.downstream.train \
  --archive /absolute/path/to/surface.npz \
  --information /absolute/path/to/information_pinn_shards \
  --training-mode joint --initialization fresh --regularization full \
  --model neural_ode --bridge latent --representation climate_manifold \
  --manifold-dim 64 --manifold-hidden-dim 512 --hidden-dim 128 \
  --history-steps 6 --history-stride 4 --horizon-steps 20 \
  --pinn --pinn-levels 500 850 --anchor none \
  --output runs/joint_neural_ode.pt --epochs 20 --batch-size 2 --device cuda

python -m climate_manifold.downstream.evaluate \
  --checkpoint runs/joint_neural_ode.pt \
  --archive /absolute/path/to/surface.npz \
  --information /absolute/path/to/information_pinn_shards \
  --split validation --output runs/joint_neural_ode.validation.json --device cuda
```

PINN가 필요 없으면 `--pinn --pinn-levels 500 850`를 빼고, surface-only 실험이면
`--information`도 뺍니다. Mode는 information 제공 여부로 결정하며 `--mode surface|enriched`로
명시할 수도 있습니다. 기본 6시간 자료에서 history_stride 4는 과거 6장을 **24시간
간격**으로 입력한다는 뜻입니다. 출력 20장은 6시간 간격의 120시간 예측입니다.

기존 A 가중치로 시작하여 함께 미세조정하려면 동일 명령에 다음을 적용합니다:

```bash
--a-checkpoint /absolute/path/to/manifold.pt --initialization pretrained
```

`--a-checkpoint`만 제공하고 `--initialization fresh`를 유지하면 그 checkpoint의 데이터·구조
계약을 쓰되 표현 가중치는 새로 초기화합니다. `pretrained`는 선택적 초기화이며 freezing을
뜻하지 않습니다. Raw 기준선에는 latent E/D 예측 경로가 없으므로 표현 공동 학습을 주장하지 않습니다.

## 저장과 적용

학습은 `train`, checkpoint 선택은 `calibration`의 **공통 미래 state MSE**를 씁니다.
보조 loss가 작아졌다는 이유로 한 비교군에 유리한 checkpoint를 선택하지 않습니다.
`validation`에서 개발 평가 후 설정을 확정하고 마지막에 `test`를 평가합니다.
기상장·정보 정규화는 train 자료에서만 적합하며 추론 때 고정합니다. Fresh 모델의 latent는
처음부터 identity 좌표를 사용하고, pretrained 모델은 기존 latent mean/scale을 고정해서
사용합니다. 학습 후 다시 latent 좌표를 바꾸어 예측기와 decoder의 좌표 계약을 깨지 않습니다.

미래 surface/상층 정보는 손실의 목표로만 사용합니다. 추론은 관측 history와 origin 정보로
시작하며 미래 정답을 입력하지 않습니다. 학습된 E/F/D, 정보 head 및 정규화/데이터 계약을
하나의 예측 checkpoint와 manifest에 저장하므로 원래 A 파일 없이 재로드할 수 있습니다.
비교군마다 잠재 좌표가 달라질 수 있어 우열은 원래 기상장 단위 RMSE/ACC로 판단합니다.

기존 frozen 결과는 `--training-mode frozen --initialization pretrained`로 명시적으로
재현할 수 있습니다. 새 joint 학습의 대조 실험이며 기본 주실험은 아닙니다.
ClimODE는 현재 `E → D → ClimODE`의 **격자 입력 공동 학습**만 지원합니다.
전역 latent 벡터에 원본 ClimODE의 격자 미분식을 적용하지 않으며 `bridge=latent`는 거부합니다.
