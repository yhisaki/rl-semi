# DDPG

```{math}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
\newcommand{\E}{{\mathrm E}}
\newcommand{\underE}[2]{\underset{\begin{subarray}{c}#1 \end{subarray}}{\E}\left[ #2 \right]}
\newcommand{\Epi}[1]{\underset{\begin{subarray}{c}\tau \sim \pi \end{subarray}}{\E}\left[ #1 \right]}
```

## 概要

前回は最適行動価値関数を推定することにより，最適方策を得る Q-learning を紹介しました．

Q-learning は強化学習の基礎となるアルゴリズムの 1 つですが，
現実の複雑なシステムに適用するには様々な課題があります．

その課題の一つに，状態空間や行動空間が実数ベクトルである問題への適用が困難であることです．
空間が離散的である場合は，
行動価値関数などをベクトル形式で保持しておけますが，
連続だと単純にベクトル形式には出来ません．
連続な値を離散化することにより，
ベクトル形式で関数を保存することが出来ますが，
状態や行動の次元に伴い，
ベクトルのサイズが指数的に増加してしまいます．

この問題の解決法は関数近似を行うことです．
いままでは行動価値関数$q: |\mathcal{S}| \times |\mathcal{A}| \rightarrow \mathbb{R}$を$q \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}$，
方策(ここでは決定的方策)を$\mu: \mathcal{S} \rightarrow \mathcal{A}$を$\mu \in \mathcal{A}^{|\mathcal{S}|}$，
という行列で捉えていたのに対して$q_\phi(s,a), \mu_\theta(s)$という，あるパラメータ$\phi,\theta$でパラメトライズされた関数として捉えます．
そうすることにより，状態空間や行動空間が実数ベクトルである問題への対処が可能となります．

しかし，関数近似を用いることにり，各行列の要素を更新する Q-learning のアルゴリズムを直接用いることは出来なくなってしまいます．その代わりに，最適行動価値関数$Q^*$や最適方策$\mu^*$をよく近似するようなパラメータ$\phi, \theta$を求めるアルゴリズムを考えなくてはなりません．

本章では，ニューラルネットワークを用いた関数近似により，強化学習を行うアルゴリズムの一つである，[DDPG](https://arxiv.org/abs/1509.02971)を紹介します．

## 行動価値関数の更新

まず最初に，Q-learning と同様，
以下の方程式を満たす関数$q_\phi$を見つけることを考えます．

$$q_{\phi} = \Upsilon_*(q_{\phi}) = r(s,a) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) \max_{a' \in \mathcal{A}} q_{\phi}(s',a') , \forall s \in \mathcal{S}, a \in \mathcal{A}$$

しかし，これは以下の２つの点で不可能です．

1. Q-learning と同様，環境の完全な情報を得ることが出来ないこと，
2. この式を完全に満たす$\phi$が存在する保証が無いことです．

特に 2 つ目に関して，この問題は，この式に限ったことではなく，
パラメトライズされた関数にと作用素の間で当然発生する問題です．
例えば関数$f_\phi(x) = \phi x$とし，作用素$T(f) = (f(x))^2 + 1$とします．
そうすると$f = T(f)$は$\phi x = (\phi x)^2 + 1$となりますが，
これを満たす定数パラメータ$\phi$は存在しません．
$q_{\phi} = \Upsilon_*(q_{\phi})$という式に対しても，同様のことが言えます．

そこで，求めるべきものを以下の評価関数を最小にする$\phi$だとします．

$$
L(\phi;\pi) = \underE{\pi}{\left( q_\phi(s,a) - \Upsilon_*(q_{\phi})(s,a) \right)^2}
$$ (q_ideal_update)

ここで$L(\phi;\pi)$は **Mean Square Bellman Error; MEBE** と呼ばれ，ある行動方策$\pi$上で得られる経験上での$q_\phi(s,a)$と$\Upsilon_*(q_{\phi_{t}})(s,a)$の平均二乗誤差を表しています．
この更新により，$q = \Upsilon_*(q)$の解に近いパラメータを得ることができます．
しかし，環境の完全な情報を得られないので，$\Upsilon_*(q_{\phi})(s,a)$に関して，実際に得られる経験$(s,a,s',r)$から，

$$
\Upsilon_*(q_{\phi})(s,a) \simeq r + \gamma \max_{a'} q_{\phi}(s',a') = y(r,s')
$$

と近似します．
また，ある行動方策$\pi$上で得られる経験という部分を，
学習を開始してから現在までの全経験$\mathcal{D}$とすると，式{eq}`q_ideal_update`の更新は以下のように書き換えられます．

$$
L(\phi;\mathcal{D}) = \underE{(s,a,s',r) \sim \mathcal{D}}{\left( q_\phi(s,a) - y(r,s')\right)^2}
$$ (q_eval_func)

ここで，学習を開始してから現在までの全経験$\mathcal{D}$を保存したバッファーは **Replay Buffer** と呼ばれます．
ただし，実際の学習においては全経験では無く，Replay Bufferから一様ランダムにサンプルされた経験のミニバッチ$\mathcal{B}$が用いられます．
また，一度の更新で式{eq}`q_eval_func`の最小値を求めることは困難であるため，
実際には，勾配ベースの手法で$\phi_t$を1ステップ更新するのみとなります．

(target_network)=
## ターゲットネットワーク

式{eq}`q_eval_func`の最小化に関して

$$y(r,s')=r + \gamma \max_{a'} q_{\phi}(s',a')$$ (q_target)

はターゲットと呼ばれており，教師信号のような意味を持っています．
しかし，ターゲットが最適化対象のパラメータである$\phi$に依存してしまっており，これは学習を不安定にしてしまいます．
そこで，解決策として$y$の計算に関しては$\phi$ではなく，$\phi$より少し遅れたネットワーク$\phi_{\text targ}$を用います．

DDPGでは，毎更新ステップにおいて，1に近いパラメータ$\rho$に対して

$$
\phi_{\text targ} \leftarrow \rho \phi_{\text targ} + (1 - \rho)\phi
$$ (delay_update)

と更新することにより，$\phi$に近いけど少し遅れたネットワークというものを実現しています．

これを用いて式{eq}`q_target`のターゲットは，

$$y(r,s')=r + \gamma \max_{a'} q_{\phi_{\text targ}}(s',a')$$ (q_target2)

となります．

## 方策の更新

$q_\phi$の更新で$\max_{a} q_\phi(s,a)$を計算する際や，経験を集める行動方策を与える際，

$$
\mu(s) = \argmax_a q_\phi(s,a)
$$

を求めることが必要となります．

しかし，$q$はニューラルネットワークで構成されており，
各状態$s$に対して，このニューラルネットワークを最適化するのは現実的ではありません．

そこで，方策を$\mu_\theta(s)$とニューラルネットワークを用いてパラメータ$\theta$でパラメトライズし，
以下の関数を最大化します．

$$
\max_{\theta} \underE{s\sim\mathcal{D}}{q_\phi(s,\mu_\theta(s))}
$$ (policy_eval_func)

実際の更新に関しては$\phi$の更新と同様，ミニバッチ$\mathcal{B}$を用いて，勾配ベースの手法で$\theta$を1ステップ更新するのみとなります
こうすることにより，$q_\phi$を最大化する方策$\mu_\theta$を近似的に得られます．

```{note}
$q$の更新における式{eq}`q_target2`のターゲットは$\mu_\theta$を用いて，

$$y(r,s')=r + \gamma  q_{\phi_{\text targ}}(s',\mu_\theta(s'))$$

と書けます．しかし，実際には方策のパラメータ$\theta$にもターゲットネットワーク$\theta_{\text targ}$が存在し，

$$y(r,s')=r + \gamma q_{\phi_{\text targ}}(s',\mu_{\theta_{\text targ}}(s'))$$

となります．$\theta_{\text targ}$は$\phi_{\text targ}$と同様，各更新ステップにおいて，

$$
\theta_{\text targ} \leftarrow \rho \theta_{\text targ} + (1 - \rho)\theta
$$

と更新されます．

```

(ddpg_behavior_policy)=
## 行動方策

Q-learningと同様，行動方策には探索と活用のバランスが大事になります．
Q-learningでは$\epsilon$-greedy法を使っていたのに対して，
DDPGでは行動空間が連続なので$\mu(s)+\epsilon$とノイズを加算することにより，探索と活用のバランスをとります．
DDPGの原論文では[OU Noise](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)という特殊なノイズを使っているのですが，
後の研究から普通のガウシアンノイズを用いても良いことがわかっている([ここを参照](https://spinningup.openai.com/en/latest/algorithms/ddpg.html#exploration-vs-exploitation))らしいです．
本書でも，ガウシアンノイズを用います．

## まとめ

以上より

```{prf:algorithm} DDPG
:label: ddpg

**Inputs** $q,\mu$の初期パラメータ$\phi,\theta$，空のReplay Buffer$\mathcal{D}$

ターゲットネットワークを初期化 $\phi_{\text targ}\leftarrow \phi, \theta_{\text targ}\leftarrow \theta$
  
**while *still time to train*:**

  1. 行動$A_t = \mu(s) + \epsilon, \ \epsilon \sim \mathcal{N}$を選択．

  2. 行動$A_t$を実行して，経験$(S_t,A_t,S_{t+1},R_t)$を得る．

  3. $(S_t,A_t,S_{t+1},R_t)$をReplay Buffer$\mathcal{D}$に蓄積

  4. **if** 更新するのであれば，以下の処理を更新回数分繰り返す

      1. Replay Buffer $\mathcal{D}$からミニバッチ$\mathcal{B}$を一様ランダムにサンプル

      2. ターゲットを計算

          $$y(r,s') = r + \gamma q_{\phi_{\text{targ}}}(s', \mu_{\theta_{\text{targ}}}(s'))$$

      3. $q$のパラメータ$\phi$を以下の勾配を用いて勾配法で1ステップ更新

          $$\nabla_{\phi} \frac{1}{|B|}\sum_{(s,a,r,s') \in B} \left( q_{\phi}(s,a) - y(r,s') \right)^2$$

      4. $\mu$のパラメータ$\theta$を以下の勾配を用いて勾配法で1ステップ更新

          $$\nabla_{\theta} \frac{1}{|B|}\sum_{s \in B}q_{\phi}(s, \mu_{\theta}(s))$$
      
      5. ターゲットネットワークを更新

          $$\phi_{\text{targ}} &\leftarrow \rho \phi_{\text{targ}} + (1-\rho) \phi \\
            \theta_{\text{targ}} &\leftarrow \rho \theta_{\text{targ}} + (1-\rho) \theta
          $$

```
