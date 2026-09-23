# Climate Manifold A의 Hybrid PINN

원본 `A_HYBRID_PINN_MANUAL.md`의 A 물리 설명을 독립 저장소에 맞게 발췌·정리했습니다.
원본 commit은 [SOURCE.md](../SOURCE.md)를 참조하세요. 실제 실행 명령은 [README](../README.md)에 있습니다.
설계 근거인 사용자 제공 방정식 문서의 원본 SHA-256은
`748f1b83972f725179a735564c467f6c6261cd780d3c90b3c8d9f49478696fc1`입니다.
해당 원문 자체는 이번 코드 추출에 포함되지 않습니다.

## 1. 필요한 실제 데이터

Z500/Z850와 10m 바람·2m 온도만으로 상층 운동량식을 계산하지 않는다.
선택한 **동일 기압면의 u/v/T/Z/omega**와 실제 지표기압 `sp`가 모두 필요하다.

| 구분 | PINN 입력과 단위 |
|---|---|
| 기본 conditioning 유지 | `z850,z500,z250,u850,v850,terrain_height,terrain_slope` |
| 기본 PINN 기압면 | 500·850 hPa; `--pinn-levels 500 850` |
| 각 기압면 추가/확인 | `u,v`: m/s, `t`: K, `z`: 지위고도 m, `w`: omega Pa/s |
| 실제 지표기압 | `sp`: Pa; 지하 기압면 및 미분 stencil 제외에 사용. `msl`로 대체하지 않음 |
| 정적 지형 | `terrain_height`: m, `terrain_slope`: 무차원. 기존 conditioning·static L2 유지 |
| 선택 확장 | `--pinn-levels 250 500 850`; 세 면 모두 u/v/T/Z/omega 필요 |
| 습도 | 기본 다운로드에 없음. 수동 제공 시 선택한 모든 기압면의 `q`: kg/kg 필요 |

ERA5 원본 `z`는 geopotential일 수 있다. 준비 단계는 선언된 단위를 검사해
`m²/s² ÷ 9.80665 → m`로 정규화하고, PDE 모듈이 **한 번만** `Phi=gZ`로 복원한다.
`w`는 기하학적 수직속도 m/s가 아니다. 단위가 m/s인 값을 omega로 받아들이지 않는다.

기존 필수 변수만 가진 INFO는 PINN 입력으로 부족하다. **새 INFO 파일/디렉터리**를 만들고
surface archive와 동일한 6h UTC 시각·격자를 사용한다. 결측/단위 모호함/시간 불일치를
0 대입이나 시간 보간으로 숨기지 않는다. 준비기는 기존과 같이 fully observed 자료를 요구하며,
PDE mask는 유효 자료 내부의 지하 기압면과 극점 주변 미분 stencil을 제외한다.

## 2. 실제 A 계산과 gradient

관측 origin의 surface와 정보로 `z0=E(surface0)+I(C0)`를 만들고,
기존 latent drift `b(z0)`로 다음 상태를 계산한다.

$$
z_1=z_0+\frac{\Delta t_{\rm hours}}{24}b(z_0),\qquad
\widehat C_0=D_{\rm info}(z_0),\quad
\widehat C_1=D_{\rm info}(z_1).
$$

정보 복원값을 train 통계로 역정규화하고, 시간차분은 **초 단위**로 계산한다.
공간 PDE는 두 복원 endpoint의 중간값에서 평가한다.
따라서 PINN gradient가 정보 decoder뿐 아니라 surface/information encoder와
latent drift까지 전달된다. 별도 surface tendency 감독이 surface decoder도 연결한다.
미래 관측 정보는 tendency 정답과 mask로만 사용하며 생성 예측의 conditioning을 갱신하지 않는다.

이 구현의 PINN은 **origin→+6h의 deterministic A drift**를 제약한다.
기존 A residual-FM의 120h member 경로에는 기존 CRPS·transition·trajectory 손실을 유지하지만,
모든 member/모든 시각에 PDE residual을 적용한 구현은 아니다.
Flow Matching의 생성 시간 `tau` 미분과 실제 기상 시간 미분을 구분한다.

## 3. 구현된 방정식과 범위

경도 `lambda`, 위도 `phi`는 radian, 기압 `p`는 Pa다.
`a=6371000 m`, `f=2 Omega sin(phi)`, `kappa=Rd/cp`로 두고,

$$
\mathcal A(Y)=\frac{u}{a\cos\phi}\partial_\lambda Y
 +\frac{v}{a}\partial_\phi Y+\omega\partial_pY
$$

를 사용한다. 학습 closure를 각각 `C_u,C_v,C_T`라 하면 잔차는 다음과 같다.

$$
\begin{aligned}
R_u&=\frac{\widehat u_1-\widehat u_0}{\Delta t}
 +\mathcal A(u)-\left(f+\frac{u\tan\phi}{a}\right)v
 +\frac{\partial_\lambda\Phi}{a\cos\phi}-C_u,\\
R_v&=\frac{\widehat v_1-\widehat v_0}{\Delta t}
 +\mathcal A(v)+fu+\frac{u^2\tan\phi}{a}
 +\frac{\partial_\phi\Phi}{a}-C_v,\\
R_T&=\frac{\widehat T_1-\widehat T_0}{\Delta t}
 +\mathcal A(T)-\kappa\frac{T}{p}\omega-C_T,\\
R_{\rm cont}&=\frac{\partial_\lambda u+\partial_\phi(v\cos\phi)}{a\cos\phi}
 +\partial_p\omega.
\end{aligned}
$$

두 인접 기압면 `p_i < p_(i+1)`의 정역학 일관성은 두께식으로 제약한다.

$$
R_{\rm thick}=Z_i-Z_{i+1}
 -\frac{R_d}{g}\frac{T_{v,i}+T_{v,i+1}}2
 \log\frac{p_{i+1}}{p_i}.
$$

습도가 있으면 `Tv=T(1+0.61q)`, 없으면 `Tv=T`인 건조 근사다.
위 식은 log-pressure 적분의 endpoint 사다리꼴 근사이며,
두 면만으로 상세한 연직구조나 정확한 층평균 온도를 복원했다는 의미는 아니다.

| 항목 | 구현/한계 |
|---|---|
| 수평 운동량 | 수평·연직 이류, Coriolis, 구면 곡률, geopotential gradient 포함 |
| 온도 | 수평·연직 이류와 압축 가열 포함; 미해상 가열 등은 closure가 보완 |
| 연속·정역학 | 기압좌표 연속식과 인접층 두께식. 전층 질량 보존 보장은 아님 |
| 미분 | 전지구 경도 주기 중앙차분, 위도/기압 좌표 간격을 사용하는 유한차분 |
| mask | 두 관측 endpoint의 `sp`와 이웃 stencil 기준. 모델이 예측한 sp로 loss를 회피하지 않음 |
| 지하층 | 선택한 전체 기압면과 이웃 stencil이 지상에 있는 칸만 사용하는 보수적 mask |
| closure | raw latent에서 u/v/T의 작은 MLP forcing; 마지막 층 0 초기화와 크기 L2 penalty |
| 정적 지형 | 기존 입력·복원 보존. 새 지형 상승류 경계식은 적용하지 않음 |

현재 포함하지 않은 항: `sp` 전층 질량 수지, 지표 omega 경계조건, 지형 상승류,
별도 thermal-wind residual, 수증기 예후식, 잠열·복사·에너지 수지.
드문 기압면 자료로 전층 적분이나 임의의 상단 omega 경계조건을 가정하지 않는다.
이 구현은 완전한 primitive-equation solver나 엄밀한 보존형 기후모델이 아니다.

## 4. Loss와 학습 단계

SI 단위 residual을 고정된 특성 크기로 나눈 뒤 유효 mask와 격자 면적으로 가중한다.
기본 크기는 운동량 `1e-3 m/s²`, 온도 `1e-4 K/s`, 연속식 `1e-5 s⁻¹`, 두께 `100 m`다.
이는 검증된 최적 계수가 아니다. `pinn_closure_normalized_rms`는 무차원 값이며,
u/v/T 각각의 residual·closure·예측 tendency·관측 tendency RMS도 SI 단위로 별도 기록한다.
무차원 loss 감소와 실제 변화량 개선을 함께 확인한다.

$$
L_{\rm PINN}=L_{\rm mom}+L_T+0.1L_{\rm cont}
 +0.1L_{\rm thick}+0.01L_{\rm closure}
 +0.1L_{\rm upper\ tendency}+0.1L_{\rm surface\ tendency}.
$$

상층 tendency 항은 선택된 u/v/T/Z/omega와 sp의 **관측 6h 변화량**을 감독한다.
이는 PDE residual만 줄이면서 정지장으로 수렴하는 문제를 견제한다.
기존 state/info/static reconstruction, decoded delta/drift, 분포·trajectory 손실은
warmup 후 기존 curriculum에 따라 함께 학습한다. 기존 loss의 `physics`와 새 `pinn_*`는 별도 항이다.

| 단계 | 동작 |
|---|---|
| A warmup, 기본 1 epoch | 관측 endpoint로 residual을 계산해 closure만 학습; 나머지 A 모듈 동결 |
| A 본 학습 | 기존 6단계 curriculum 재개 + 복원장 PINN. encoder/decoder/drift/closure 함께 학습 |
| PINN ramp | warmup 뒤 3 epoch 동안 전체 가중치 `0.1 × min(1, joint_epoch/3)` |
| Best A 선택 | warmup·curriculum·ramp 완료 후 expert_validation. 기존 선택 점수에 plateau PINN 가중치를 더함 |

`--epochs`에는 warmup이 포함된다. `process` profile에서 필요한 최소치는
`warmup_epochs + max(5 × curriculum_interval + 1, ramp_epochs)`다.
이 저장소의 trainer/runner 기본 interval 4에서는 최소 **22 epoch**다.
기본 60 epochs는 최적값을 보장하지 않는다.
