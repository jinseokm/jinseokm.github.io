---
title: "ML with Pytorch 01"
categories:
  - Study
tags: [AI, Pytorch, Machine Learning]
toc: true
toc_sticky: true
toc_label: "On this page"
published: false
use_math: true
---

## 선형회귀 (Linear Regression)
선형회귀란 어떠한 현상을 가장 잘 나타내는 함수식을 찾는 과정이라고 생각하시면 됩니다. 이러한 함수식을 가설(Hypothesis)이라고 하며, 주로 $$h(x)$$로 나타냅니다. 

$$
\begin{equation}
h(x) = wx
\end{equation}
$$

$$h(x) = wx$$라는 간단한 가설을 세웠습니다. 이 식을 풀어서 정의하자면 $$y$$에 대해서 $$x$$가 가지는 영향력을 $$w$$라는 가중치로 나타낸 것과 같습니다. 여기서 $$x=1, y=2$$인 무척 단순한 경우를 생각해보겠습니다. 우리는 $$2=w*1$$ 에서, $$w$$의 값이 2라는 사실을 알지만, 컴퓨터는 이를 알지 못합니다. 즉, 우리는 $$w$$의 값을 2에 보다 가깝게 맞춰주는 `최적화` 작업을 하게 되고, 이 과정이 바로 `학습` 이라고 할 수 있습니다.

먼저, 앞서 정의한 $$h(x)$$에 대해서, 가중치 $$w$$의 초기값이 0인 경우를 생각해보겠습니다. 실제값은 2인데, $$wx=0 \cdot 1=0$$으로 예측값과 차이가 발생했습니다. 실제값과 예측값의 차이를 통해, 가중치를 어떻게 해줘야 이 차이를 줄일 수 있을지를 생각합니다. 여기서 사용되는 개념이 `비용함수`, `손실함수` 등의 개념입니다. 비용함수를 다음과 같이 정의해보겠습니다. 비용함수를 적용하면 cost는 4가 나오게 되고, cost를 최소화 하는 것을 통해 가중치를 2에 맞춰가는 최적화를 수행하게 됩니다.

$$
\begin{equation}
C(w) = (wx-y)^2
\end{equation}
$$


## Gradient Descent Algorithm
그렇다면, 이런 최적화는 어떻게 할 수 있을까요? 여기서 `경사하강법 Gradient Descent Algorithm` 이 등장합니다. 경사하강법은 머신러닝에서 가장 먼저 공부해야 하고, 또 기본 바탕이 되는 개념으로 확실하게 알아야 할 필요가 있습니다. 경사하강법에서는 함수식에서 기울기가 완만해지는 방향으로 최적화를 수행합니다. $$w$$의 업데이트를 수식으로 나타내면 다음과 같습니다. 비용함수에 대해 해당 지점에서의 기울기를 이용해 업데이트를 수행하게 됩니다. 기울기는 미분을 통해 얻어냅니다. $$\eta$$는 학습률을 나타내는데, 이는 한번의 계산으로 얼만큼 갱신할지를 정하는 파라미터입니다. 이 예제에서는 0.05로 설정하겠습니다.

$$
\begin{equation}
w \gets w - \eta \frac{\partial C}{\partial w}
\end{equation}
$$

$$
\begin{equation}
\frac{\partial C}{\partial w} = 2(wx-y)x
\end{equation}
$$
미분을 통해, $$w$$가 0일때 얻어진 기울기는 -4로, 다음 $$w$$의 값은 $$w=0-0.05*(-4)$$가 됩니다.
<center>
<figure style="width:70%"> <img src="/Images/Study/mlstudy/cost.jpg" alt=""/>
<figcaption>가중치와 비용의 관계</figcaption>
</figure>
</center>