# Dynamics encoder v2: 관측 전이로 제약한 표현

브랜치: `feature/dynamics-transition-v2`.
`--dynamics-version v1|v2`로 선택합니다. 기본은 v1입니다.
ConvGRU/Temporal Conv에 실제 학습 구간의 전이 이웃을 이용하는 보조 감독을 추가합니다.
v2의 기본 입력은 지표 history와 z500·z850 history 및 고정 지형 고도입니다.
`latent`와 `guided`, T1 OFF/ON,
기존 `field / field_latent / vector_only` 모두와 함께 사용할 수 있습니다.

## v2의 입력 정보

| 정보 | Encoder 사용 | 전이 관계에서의 역할 |
|---|---|---|
| `msl`, `t2m`, `u10`, `v10` | 관측 history와 시간 차분 | 현재 유사도·이력·후속 변화 |
| `z500`, `z850` | 각 시각의 실제 관측값과 시간 차분 | 지표장과 함께 현재 유사도·이력·후속 변화 |
| `terrain_height` | 시간에 고정된 공간 조건 | E_d가 지형에 조건화된 표현을 생성 |

`z500/z850`는 각각 500/850 hPa 면의 **지위고도**입니다. ERA5의 geopotential을
전처리에서 중력가속도로 나누어 m 단위로 통일합니다. 원래 지표 변수는 `u10/v10`이며
`u100/v100`으로 바꾸지 않습니다. 최종 예측 변수는 기존 네 지표 변수를 유지합니다.

지위고도 입력은 현재 한 장을 history 전체에 복제하지 않고, 해당 관측 시각의
sidecar에서 읽습니다. 지형은 고정장으로 넣고 시간 차분이나 미래 변화 정답으로
취급하지 않습니다. 같은 전구 격자의 지형은 모든 날짜에 같으므로 사례 사이의
시간적 유사도 거리에 별도 차이를 만들지 않습니다.

`--dynamics-information geopotential_terrain`이 v2 CLI의 기본이며,
`--dynamics-information off`로 원래 지표 history만 사용하는 v2도 선택할 수 있습니다.
v1은 계속 off입니다. 지위고도·지형을 켜면 E_d 입력 stem의 채널 수가 증가하지만
latent 출력 크기와 최종 예측 모델의 인터페이스는 유지됩니다. 이를 끈 v2는 v1과
같은 Encoder 크기로 전이 감독 효과를 분리할 수 있습니다.

정규화는 기존 train-only 변수별 면적 통계를 사용하며 지형의 공간 차이를 보존합니다.
필수 sidecar 변수가 없거나 단위·격자·시간·static 선언이 맞지 않으면 오류로 처리합니다.

## 무엇을 학습하는가

한 사례를 관측 이력 H_i, 현재 장 A_i, 미래 변화 ΔX_i로 나눕니다.
6일 관측 설정에서는 H_i가 현재까지의 6일 history이며, 미래 변화는 학습 정답입니다.

1. 실제 train loader가 사용하는 window만 후보 bank에 넣습니다.
2. 현재 A_i와 가까운 후보 A_j를 찾습니다. 이 검색에는 미래를 사용하지 않습니다.
3. 각 후보의 관측 이력 H_j를 같은 E_d로 encode하여 가중 empirical measure를 만듭니다.
4. 이력과 이후 변화가 비슷한 사례를 latent에서도 구별하여 표현하도록 관계 손실을 줍니다.

현재 장의 거리로 prior를 정합니다.

\[
w_{ij}=\operatorname{softmax}_j(-d_A(A_i,A_j)/\tau_A),\qquad
\mu_i=\sum_j w_{ij}\delta_{E_d(H_j)}.
\]

`mu_i`는 여러 support와 가중치를 유지합니다. 가중 평균 latent 하나로 바꾸어
그 점에 모든 과거를 맞추는 MSE는 사용하지 않습니다. 후보 latent는 현재 E_d로
다시 계산하며 stop-gradient로 사용합니다. 오래된 latent cache는 사용하지 않습니다.

관측된 과거 이력·후속 변화로 감독 분포 p를, latent 거리로 모델 분포 q를 정의합니다.

\[
r_{ij}=\alpha d_H(H_i,H_j)+\beta d_\Delta(\Delta X_i,\Delta X_j),
\]
\[
p_{ij}=\operatorname{softmax}_j(\log w_{ij}-r_{ij}/\tau_R),\qquad
q_{ij}=\operatorname{softmax}_j(\log w_{ij}-d_z(z_i,z_j)/\tau_z).
\]
\[
L_{v2}=\lambda\left[\mathrm{KL}(p_i\Vert q_i)
+\gamma\sum_j w_{ij}\bigl(b(d_z(z_i,z_j))-b(r_{ij})\bigr)^2\right],
\quad b(u)=u/(1+u).
\]

두 번째 항은 관측 관계와 latent 거리의 크기도 연결하는 선택형 보조 항입니다.
`--transition-geometry-weight 0`이면 관계 KL만 사용합니다. 공간 복원 통계의
`--statistical-loss w2|kl_entropy|signed_measure`와 이 관계 KL은 별개입니다.

거리 계산은 기존 train normalization과 지리적 위치를 유지하는 면적 가중 축소장을
사용하며 정보 모드가 켜지면 여섯 동적 채널(지표 4 + 지위고도 2)을 함께 비교합니다.
이력은 현재로부터의 상대 변화, 미래는 현재 대비 변화/물리적 경과 일수로
구성합니다. 현재가 비슷해도 이력과 후속 변화가 다르면 동일 latent로 모을 필요가 없습니다.

## 누출 방지와 계산 범위

- bank는 `train`에서만 생성하며 window stride와 max_windows도 그대로 적용합니다.
- 후보가 많으면 결정론적으로 최대 bank_size개를 선택합니다. 기본 4096개입니다.
- 자기 자신뿐 아니라 **관측부터 미래 정답까지 전체 시간 구간이 겹치는 사례**를 제외합니다.
- 후보 선택과 prior는 현재 장만 사용합니다. 미래는 detached 감독 분포를 만드는 데만 씁니다.
- 미래 지위고도도 보조 관계/latent 정답에만 사용하며 예측 입력으로 전달하지 않습니다.
- 미래 변화의 범위는 주예측의 실제 학습 horizon으로 제한합니다. 원자료 window에 더 먼
  미래가 담겨 있어도 이 보조 정답에 사용하지 않습니다.
- 최소 두 유효 후보가 필요합니다. 자료가 너무 짧으면 조용히 무시하지 않고 오류로 알립니다.
- v2 손실은 training에서만 계산합니다. 모델 선택은 기존 calibration의 주예측 MSE입니다.
- checkpoint에는 설정과 bank 출처가 기록됩니다. 추론은 bank 없이 기존 E_d와 M으로 수행합니다.

이는 물리 법칙으로 역산한 manifold나 완전한 조건부 확률 밀도 모델이 아닙니다.
관측 자료가 제공하는 시간 전이 관계를 학습하는 첫 구현입니다. 앙상블 샘플러를 새로
만들지 않으며, 다양성 보존이나 물리적 적합성을 보장하지 않습니다. 기존 예측·공간·
동역학 감독을 유지하고 KL, 관계 거리, latent variance, 이웃 유효 수를 함께 확인합니다.
기본 4×8 검색 descriptor는 거친 패턴을 사용하므로 국소 태풍 구조를 충분히
구별하는지는 별도 검증해야 합니다.

## 실행 옵션

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `--dynamics-version` | `v1` | v2 보조 감독 선택 |
| `--dynamics-information` | v2: `geopotential_terrain`, v1: `off` | 지위고도 history·지형 조건 선택 |
| `--transition-weight` | 0.1 | 전체 v2 손실 가중치 |
| `--transition-neighbors` | 8 | 현재 유사 사례 수 |
| `--transition-bank-size` | 4096 | train 후보 상한 |
| `--transition-endpoint-temperature` | 1 | 현재 유사도 prior의 온도 |
| `--transition-relation-temperature` | 1 | 관측 관계 분포의 온도 |
| `--transition-latent-temperature` | 1 | latent 관계 분포의 온도 |
| `--transition-history-weight` | 1 | 과거 이력 거리 가중치 |
| `--transition-future-weight` | 1 | 후속 변화 거리 가중치 |
| `--transition-geometry-weight` | 0.1 | 유계 거리 정합 항 |
| `--transition-pool-lat`, `--transition-pool-lon` | 4, 8 | 검색 descriptor 격자 |
| `--transition-exclusion-steps` | 0 | 전체 window 겹침 금지에 추가할 archive step |

값은 v2일 때 해석합니다. 직접 CLI에서 v1과 transition 전용 옵션을 함께 주면
설정 실수를 막기 위해 거절합니다. shell runner는 raw/공간만 조건에 이 옵션을 전달하지 않습니다.

기존 전체 명령에서 clone branch를 `feature/dynamics-transition-v2`로 변경하고,
실행 직전에 아래를 넣으면 같은 12조건 비교에서 Dynamics 조건에만 v2가 적용됩니다.

```bash
export DYNAMICS_VERSION=v2
export DYNAMICS_INFORMATION=geopotential_terrain
export TRANSITION_WEIGHT=0.1 TRANSITION_NEIGHBORS=8 TRANSITION_BANK_SIZE=4096
export TRANSITION_HISTORY_WEIGHT=1 TRANSITION_FUTURE_WEIGHT=1
export TRANSITION_GEOMETRY_WEIGHT=0.1
bash scripts/run_full_manifold_matrix.sh
```

v1로 돌아갈 때는 `export DYNAMICS_VERSION=v1`입니다. v2 결과 폴더는
`runs/full-v2-w2`, `runs/full-v2-kl_entropy`, `runs/full-v2-signed_measure`이며
통합 표는 `full-matrix-v2-summary.csv`입니다. 기존 v1 결과 폴더와 분리합니다.
`DYNAMICS_INFORMATION=off`이면 `runs/full-v2-surface-{loss}`와
`full-matrix-v2-surface-summary.csv`로 추가 정보를 사용한 실험과도 분리됩니다.
ERA5 원본·작업 폴더·GPU 제한은 기존 [전체 실행 명령](full_manifold_matrix.md)과 같습니다.

버전별로 같은 seed·모델·통계 손실·감독을 맞춰 주예측의 RMSE/ACC와 rollout을 비교하세요.
v1/v2는 별도 suite로 보존하며, 다른 감독 조건을 같은 조건처럼 합쳐 집계하지 않습니다.
정보 모드를 켠 v2는 추가 관측 이력과 Encoder 입력 용량도 달라지므로, 전이 손실의
효과를 분리하려면 지표 history만 사용하는 v2와도 비교해야 합니다.
v2 보조 KL 감소만으로 예측력이 좋아졌다고 판정하지 않습니다.
