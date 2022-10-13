---
title: "Learning Strides in Convolutional Nerual Networks"
excerpt: "DiffStride"
categories:
  - Papers
tags: [CNN, ICLR]
toc: true
toc_sticky: true
toc_label: "On this page"
published: true
use_math: true

date: 2022-06-01
last_modified_at: 2022-06-28
---

# [Learning Strides in Convolutional Nerual Networks (ICLR 2022)](https://openreview.net/pdf?id=M752z9FKJP)
구글 리서치 팀에서 수행한 연구로, ICLR 2022에서 Outstanding Paper Awards를 받은 논문입니다. 간략하게 내용을 요약해보자면, 'feature의 downnsampling을 spatial domain이 아닌 spectral domain에서 수행하고, 그 stride를 미분가능한 함수로 정의하여 stride의 폭까지 학습할 수 있다' 라는 내용의 논문입니다.

ConvNet에서는 커널이 sliding window로 움직이며 주변과의 상호작용이 정의됩니다. 정의된 pattern 혹은 feature는 네트워크에 가중치를 가지면서 학습되고 결과적으로 MLP와 다르게 translation invariance 특성을 가질 수 있게 되었습니다. 
ConvNet은 Spatial domain에서 얼마만큼의 정보를 한 묶음으로 볼 것인가? 라는 질문과 함께 parametric study 적인 많은 연구들이 수행되어 왔습니다. 
학습/평가에 사용되는 데이터셋에 대해서 효과적인 kernel, stride, pooling 의 사이즈에 대한 연구말이죠. 논문에 의하면 ResNet-18 베이스에서 stride size를 잘못가져가는 것만으로도 네트워크의 성능이 18% 이상 떨어질 수 있다고 합니다.

Stride는 기본적으로 1, 2, ... 등의 일반적인 정수형의 크기를 시작으로, 분수형태의 stride size (Graham, 2014)를 사용하거나 spectral  domain에서 pooling을 수행하기도 했습니다 (Rippel et al., 2015). 이러한 stride의 크기 또한 하나의 파라미터이기 때문에, Network Architecture Search (Zoph & Le, 2017) 와 같은 방법론적 연구들도 수행되었습니다. 하지만 기존의 연구들에서 이런 stride를 디테일하게 학습하고 검토하기에는 다른 더 큰 부분들이 많기 때문에 현실적으로 어려움이 컸습니다.

저자들은 이러한 백그라운드를 통해서 stride를 hyperparameter가 아닌, `trainable parameter` 로 보고자 했고, 더 나아가 spectral pooling 까지 수행할 수 있는 `DiffStride` 를 제안했습니다. 단일 채널에 대한 업데이트는 다음과 같습니다. Cropping에 대해서는 미분이 불가능하기 때문에 stopping gradient를 적용했다고 합니다.

<center>
<figure style="width: 70%"> <img src="/Images/Study/diffstride/diffstride.png" alt="DiffStride"/>
<figcaption>DiffStride 단일 채널 forward/backward</figcaption>
</figure>
</center>

<center>
<figure style="width: 70%"> <img src="/Images/Study/diffstride/algorithms.png" alt="DiffStride"/>
<figcaption>DiffStride 알고리즘</figcaption>
</figure>
</center>

<center>
<figure style="width: 70%"> <img src="/Images/Study/diffstride/mask1.png" alt="DiffStride"/>
<img src="/Images/Study/diffstride/mask2.png" alt="DiffStride"/>
<figcaption>DiffStride Masking</figcaption>
</figure>
</center>

Skip connection을 활용하는 residual block에는 다음과 같이 diffstride를 적용합니다.
<center>
<figure style="width: 70%"> <img src="/Images/Study/diffstride/residual.png" alt="DiffStride"/>
<figcaption>Residual Block with DiffStride</figcaption>
</figure>
</center>

Imagenet classification task에서, 기존의 Strided Conv. 보다 훨씬 좋은 성능을 보입니다. 저자들은 이러한 결과로부터, stride 세팅이 성능에 큰 영향을 주는 것을 말하며, DiffStride는 보다 효과적인 spectral pooling이 가능하다고 말합니다.
<center>
<figure style="width: 70%"> <img src="/Images/Study/diffstride/imagenet.png" alt="diffstride"/>
<figcaption>Imagenet Classification Task with DiffStride</figcaption>
</figure>
</center>


이외에도 논문에는 오디오, 멀티태스크, CIFAR-10(100) classification task 에 대한 결과 또한 보실 수 있습니다. 또한 ResNet 뿐만 아니라 DenseNet, EfficientNet 에도 적용해보았는데, DenseNet에서는 일부 모델에서 개선이 되었고, EfficientNet은 전반적으로 개선되었다고 합니다. 이산 푸리에 변환을 통한 pooling method라는 것이 무척이나 흥미로웠고, [openreview](https://openreview.net/forum?id=M752z9FKJP)에서도 기존의 알고리즘을 대체 가능하다는 점에 높은 점수를 준 리뷰어들이 다수기도 했네요.
코드는 아쉽게도 현재는 tensorflow만 제공되고 있습니다. 전체적으로 in/out feature size를 동적으로 바꿔줘야 하는 점이 있어서 바로 적용하기에는 손볼 곳이 많을 것 같네요. 

# References
1. Riad, Rachid, et al., "Learning Strides in Convolutional Neural Networks.", International Conference on Learning Representations, 2022.