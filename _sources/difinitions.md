# Definitions

- 状態空間(State Space): $\mathcal{S} \ni s$
- 行動空間(Action Space): $\mathcal{A} \ni a$
- 状態遷移確率分布関数: $p: \mathcal{S} \times \mathcal{S} \times \mathcal{A} \rightarrow [0, 1]$:

  $p(s'|s, a) := \Pr(S_{t+1} = s'| S_{t} = s, A_t = a),\ \ \ \forall t \in \mathbb{N}_0.$

- 報酬関数(Reward Function): $r: \mathcal{S} \times \mathcal{A} \rightarrow [r_{\text{min}}, r_{\text{max}}]$
- 初期状態確率分布関数(Initial State Distribution): $p_0: \mathcal{S}  \rightarrow [0, 1]:p_0(s) := \Pr(S_0 = s)$
