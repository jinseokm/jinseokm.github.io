---
title: "Inductive Bias 에 대해서"
categories:
  - Study
tags: [AI, Pytorch, Machine Learning, CNN]
toc: true
toc_sticky: true
toc_label: "On this page"
published: true
use_math: true

date: 2022-12-21
---

Inductive bias란 학습 알고리즘에서 데이터와 독립적으로 설정한 일종의 추가적인 가정이라고 할 수 있습니다.

> An inductive bias allows a learning algorithm to prioritize one solution (or interpretation) over another, independent of the observed data (Mitchell, 1980).

Inductive bias 는 주로 복잡성-유연성에 대해서, 편향-분산 trade off 와 비슷한 개념의 tradeoff 를 갖습니다. 이상적으로 inductive bias 를 설정했다면 관측되지 않은 데이터에서도 동일한 성능을 낼 만큼의 일반화가 가능해지는 반면, 가정이 잘못되었다면 (mismatched bias) 오히려 성능이 떨어질 수 있습니다.

> Inductive biases often trade flexibility for improved sample complexity and can be understood in terms of the bias-variance tradeoff (Geman et al., 1992). Ideally, inductive biases both improve the search for solutions without substantially diminishing performance, as well as help find solutions which generalize in a desirable way; however, mismatched inductive biases can also lead to suboptimal performance by introducing constraints that are too strong.

이런 inductive bias의 가장 익숙한 예시가 바로 선형 회귀 모델에서의 `(X, Y) 의 관계가 linear 하다` 라는 가정이 아닐까 싶습니다. 이 가정하에서, 모델은 데이터가 선형적일때 강한 성능을 보이겠지만, 비선형적인 데이터에 대해서는 성능이 좋지 않을 것입니다.


<details>
<summary> Bias-Variance Tradeoff </summary>
편향(bias)은 타겟에서부터 얼마나 떨어져있는지를 나타내며, 분산(variance)은 얼마나 데이터가 퍼져있는지를 나타냅니다. 가장 좋은 것은 low bias, low variance이지만, 현실적으로 이 둘은 trade-off 관계에 있어 둘 다 낮은 경우는 보기 힘듭니다.

<figure style="width:40%"> <img src="/Images/Study/inductive-bias/bias-variance.png" alt="bias and variance"/><figcaption> Image from <a href="http://scott.fortmann-roe.com/docs/BiasVariance.html">Link</a> </figcaption> </figure>
<figure style="width:55%"> <img src="/Images/Study/inductive-bias/b-v-tradeoff.png" alt="bias and variance"/><figcaption> Bias 와 variance 는 서로 tradeoff 관계에 있다. </figcaption> </figure>
</details>

# Relational Inductive Bias

Inductive bias 는 관계성에 의존하는지에 따라 relational/non-relational 로 나뉘는데, 여기서는 relational inductive bias 만 다루기로 합니다.

> Though beyond the scope of this paper, various non-relational inductive biases are used in deep learning as ****well: for example, activation non-linearities, weight decay, dropout (Srivastava et al., 2014), batch and layer normalization (Ioffe and Szegedy, 2015; Ba et al., 2016), data augmentation, training curricula, and optimization algorithms all impose constraints on the trajectory and outcome of learning.
> 

Fully connected layer, CNN, RNN, GNN 에 먼저 정리해보자면 아래의 표와 같습니다.

<center>
<figure style="width:70%"> <img src="/Images/Study/inductive-bias/table.png" alt="inductive bias"/> </figure>
</center>

<center>
<figure style="width:70%"> <img src="/Images/Study/inductive-bias/param_share.png" alt="inductive bias"/>
<figcaption>FC, CNN, RNN. 같은 파라미터는 동일한 색으로 표현됨 </figcaption> </figure>
</center>


- (Standard) Fully Connected Layer

Input 하나하나가 출력에 대해 각각 weight 를 가지는 형태(all-to-all)입니다. 재사용되는 weight 가 없이 독립적이기 때문에 모든 input 이 모든 output 에 영향을 미치게 되며, 이로 인해 inductive bias 가 약하다고 할 수 있습니다. 또한 모든 파라미터가 독립적으로 존재하기 때문에 invariance 가 없습니다.

- Convolutional Neural Network

FC layer 와 달리, 지역적으로 파라미터를 공유합니다. 위의 그림에서 필터를 통과할 때, 동일한 필터가 이미지 전체를 훑고 지나가게 됩니다. CNN 은 좌표공간에서 인접한 데이터들일수록 강한 관계를 가지도록 설계되어 있습니다. 이러한 특성이 locality, 즉 지역성에 대한 inductive bias 를 나타냅니다. 또한 필터의 parameter sharing 을 통해 위치에 상관없이 동일한 rule 을 적용하게 됨으로써, 눈, 코, 입 등의 특정 feature 를 추출할 수 있게 됩니다. 이미지와 같이 고정된 그리드 형태의 데이터에 활용될 때 성능이 좋은 경우가 많습니다. 

- Recurrent Neural Network

Sequential data 를 처리하는 네트워크로, 이전의 은닉상태 - 현재의 입력이 관계되는 Markov dependence 를 갖고, 이 parameter/rule 은 각 step 에서 재사용됩니다. RNN 은 다음과 같은 bias 를 갖고 있습니다. 

1. The sequential processing of the input: 입력 토큰은 하나씩 순차적으로 처리됨
2. No direct access to the past tokens: 다음 토큰을 처리할 때 액세스할 수 있는 숨겨진 상태/메모리에 과거 토큰의 모든 정보를 압축해야 함
3. Recursion: 모델은 모든 time step 에서 다양한 입력에 대해 동일한 함수를 재귀적으로 적용함

즉, 일련의 sequential 한 event 의 결과는 시간에 의존하지 않고, 마르코프 구조를 통해 sequence 의 locality (Sequentiality) 를 inductive bias 로 갖습니다.

> For example, the outcome of some physical sequence of events should not depend on the time of day. RNNs also carry a bias for locality in the sequence via their Markovian structure (Table 1).
> 

- Graph Networks

노드와 엣지로 이루어지는 그래프는 몇가지의 inductive bias 를 가집니다.

1. 엔티티간의 임의의 관계를 노드와 엣지로 표현함. 엣지가 없으면 노드간에 관계가 없고 서로 영향을 주지 않음
2. Permutation invariance: 관계가 순열에 대해 무관함
3. 각 노드별, 엣지별 함수는 각각 모든 노드와 엣지에서 재사용됨

<center>
<figure style="width:70%"> <img src="/Images/Study/inductive-bias/graph_permute.png" alt="inductive bias gnn"/>
<figcaption> Permutation invariance of GNN </figcaption> </figure>
</center>


## References
1. [https://arxiv.org/abs/1806.01261](https://arxiv.org/abs/1806.01261)
2. [https://www.dacon.io/forum/405840](https://www.dacon.io/forum/405840)
3. [https://www.baeldung.com/cs/ml-inductive-bias](https://www.baeldung.com/cs/ml-inductive-bias)