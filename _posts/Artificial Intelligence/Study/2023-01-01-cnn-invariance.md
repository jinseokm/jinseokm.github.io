---
title: "CNN의 Translation Invariance에 대하여"
categories:
  - Study
tags: [AI, Pytorch, Machine Learning, CNN]
toc: true
toc_sticky: true
toc_label: "On this page"
published: true
use_math: true

---

<center>
<figure style="width:70%"> <img src="/Images/Study/cnn-translation/0.png" alt="Equivariance and Invariance"/>
</figure>
</center>

# Translation Invariance
Translation 에 invariant 하다는 것은 입력이 바뀌어도 출력은 바뀌지 않는 것을 의미합니다. 입력 이미지가 고양이인지 아닌지 판별하는 함수 $f$ 는 고양이의 위치가 바뀌어도 동일하게 고양이 라고 판별해 줄 것입니다.

<center>
<figure style="width:70%"> <img src="/Images/Study/cnn-translation/2.png" alt="Translation Equivariance"/>
</figure>
</center>


# Translation Equivariance

Translation Equivariance 는 입력이 바뀌면 출력도 바뀐다는 뜻입니다. 예를들어 고양이가 있는 영역에는 1 을 칠하고, 아닌 영역에는 0 을 칠하는 함수 $f$ 가 있다고 합시다. 이때, 고양이의 위치를 이동시키는 $S$ 함수를 이용하여 입력 이미지를 바꿔주었다면 출력도 동일하게 변경된 고양이의 위치에 칠해져 있을 것입니다.

<center>
<figure style="width:70%"> <img src="/Images/Study/cnn-translation/1.png" alt="Translation Equivariance"/>
</figure>
</center>

Convolutional layer 자체는 아래 그림과 같이, 입력이 변경되면 그에 맞춰 출력도 변경되기 때문에 translation **equivariant** 합니다. 

<center>
<figure style="width:70%"> <img src="/Images/Study/cnn-translation/1.gif" alt="Translation Equivariance"/>
</figure>
</center>

<center>
<figure style="width:70%"> <img src="/Images/Study/cnn-translation/3.jpg" alt="Translation Equivariance"/>
</figure>
</center>

그렇다면 Convolutional layer 자체는 translation equivariant 한데, 어떻게 CNN 을 사용한 네트워크들은 spatial translation **invariant** 할 수 있을까요? 인터넷을 많이 돌아다녔는데, 제가 아직 공부가 부족해서 이게 맞다 라는것을 정확히 이해하지 못한 것 같아서 관련 내용들을 적어보겠습니다.

<center>
<figure style="width:70%"> <img src="/Images/Study/cnn-translation/4.png" alt="Translation Equivariance"/>
<figcaption> 위치에 상관없이 고양이라는 정답을 출력 </figcaption>
</figure>
</center>

# Pooling: patially invariant

먼저 pooling layer 를 알 필요가 있습니다. 아래 그림은 최대값을 출력하는 max pooling layer 입니다. 

<center>
<figure style="width:70%"> <img src="/Images/Study/cnn-translation/5.png" alt="Translation Equivariance"/>
</figure>
</center>

‘ㄱ’ 모양의 데이터에 대한 필터를 적용했을 때, max pooling 을 거치면 왼쪽 상단의 데이터가 활성화 됩니다.

<center>
<figure style="width:70%"> <img src="/Images/Study/cnn-translation/1.jpg" alt="Translation Equivariance"/>
</figure>
</center>

이 데이터를 한 칸씩 아래로 시프트했을 때에도, max pooling layer 를 거치면 출력에는 변화가 생기지 않습니다.

<center>
<figure style="width:70%"> <img src="/Images/Study/cnn-translation/2.jpg" alt="Translation Equivariance"/>
</figure>
</center>

즉, 입력이 변해도 출력이 변하지 않게 되는 것입니다. Deep learning text book 에서도 이러한 내용을 찾아볼 수 있습니다. 입력값이 오른쪽으로 한 칸씩 시프트 되었으나, pooling 의 결과는 절반만 변경되었습니다. max pooling 은 주변 값들에 대해서 정확한 위치가 아닌, 최댓값에 민감하기 때문으로, 이런 특성으로 인해 부분적인 translation invariance 가 생깁니다.

<center>
<figure style="width:70%"> <img src="/Images/Study/cnn-translation/6.jpg" alt="Translation Equivariance"/>
</figure>
</center>

# FC, softmax, parameter sharing

하지만 이것만으로는 전체에 대해서 invariant 하다고 완벽하게 말하지는 못하는 것 같습니다. 찾다가 이 글이 가장 설득력 있어보여서 가져왔습니다. ([https://seoilgun.medium.com/cnn의-stationarity와-locality-610166700979](https://seoilgun.medium.com/cnn%EC%9D%98-stationarity%EC%99%80-locality-610166700979), [https://qr.ae/prNtyH](https://qr.ae/prNtyH))


<center>
<figure style="width:70%"> <img src="/Images/Study/cnn-translation/7.png" alt="Translation Equivariance"/>
</figure>
</center>

이런 식으로 (학습된) 모델이 있다고 했을 때, 다음과 같은 과정을 거치게 됩니다.

- 초록 블록: 이미지의 왼쪽 하단으로부터 눈, 코 등의 feature 를 탐지
- 노랑 블록: 눈, 코 등의 탐지된 feature 를 이용해서 얼굴 탐지
- FC layer: 얼굴 탐지 feature 활성화
- softmax 를 거쳐 human body 검출

여기서, 얼굴이 왼쪽 상단에 있는 경우를 살펴보면 아래와 같습니다. 


<center>
<figure style="width:70%"> <img src="/Images/Study/cnn-translation/8.png" alt="Translation Equivariance"/>
</figure>
</center>

위치가 바뀌었지만 결과적으로 얼굴이 탐지되면 FC layer 에서 해당 노드가 활성화되고, 그로 인해 human body 검출이 됩니다. 즉, conv layer 들을 거치는 동안 translation equivariant 하다가, 말단의 fc, softmax layer 를 거치며 invariant 한 특성을 얻게 되는 것이라고 합니다. 이에 더해, CNN 의 특성인 각 filter 의 parameter sharing 이 더해져, 위치가 달라져도 동일한 결과를 출력하는 것으로 생각됩니다.

# Data augmentation

이에 더해서, cs 231n 수업에서 제출되었던 [레포트 자료](https://arxiv.org/abs/1801.01450)에서도 관련 연구 내용이 있었는데, 이 레포트에 의하면 data augmentation 을 통한 파라미터 일반화가 가장 큰 영향을 끼친다고 합니다. 아래 그림은 transformation 에 대한 민감도로, model c 는 conv layer 만, model cp 는 conv+pooling, cpcp 는 (conv+pooling)*2 를 의미합니다.

<center>
<figure style="width:70%"> <img src="/Images/Study/cnn-translation/9.png" alt="Translation Equivariance"/>
<figcaption> by E Kauderer-Abrams </figcaption>
</figure>
</center>

# Brief test

실제로 CNN 에서 pooling layer 가 얼마나 영향을 끼치는지 알아보고자 [Divyanshu 의 포스트](https://divsoni2012.medium.com/translation-invariance-in-convolutional-neural-networks-61d9b6fa03df)를 참고했는데, 비교군이 cnn w/ w/o pooling 이 아니어서 cnn 으로 모델을 변경해서 실험을 했습니다.

- MNIST dataset, zero padding, batch_size=128, epoch=5
    - Model A: CNN w/o pooling
    - Model B: CNN w pooling
    - Model C: (CNN w/ pooling) x 2

| Models | acc: default | acc: shift min | acc: shift |
| --- | --- | --- | --- |
| A | 0.974 | 0.959 | 0.686 |
| B | 0.979 | 0.967 | 0.718 |
| C | 0.987 | 0.982 | 0.843 |

Pooling layer 유무로 결과의 차이가 있긴 하지만, mnist dataset 의 경우 이미지의 크기가 28x28 로, 그렇게 큰 데이터셋이 아니라서 일반화해서 결론을 내리기에는 어렵지 않나 싶긴 합니다.

동일한 세팅으로, data augmentation 에 대한 테스트도 수행해보았습니다.

- MNIST dataset, zero padding, batch_size=128, epoch=5, data augmentation (width +- 10%, height +- 10%)
    - Model A: CNN w/o pooling
    - Model B: CNN w pooling
    - Model C: (CNN w/ pooling) x 2

| Models | acc: default | acc: shift min | acc: shift |
| --- | --- | --- | --- |
| A | 0.968 | 0.974 | 0.950 |
| B | 0.980 | 0.982 | 0.966 |
| C | 0.989 | 0.989 | 0.983 |

확실히 data augmentation 으로 성능이 좋아진 것으로 보이긴 합니다.

# What is wrong with convolutional neural nets? by Geoffrey Hinton

극단적인 예시이긴 하지만, translation invariance 의 문제점으로 꼽히는 그림입니다. CNN 이 얼굴의 부분적인 특징을 잡기 때문에, 오른쪽 그림도 결과적으로 얼굴이라고 인식해버리는 것입니다. Hinton 은 이러한 문제를 해결하고자 캡슐 네트워크를 제안했습니다.

<center>
<figure style="width:70%"> <img src="/Images/Study/cnn-translation/10.png" alt="Translation Equivariance"/>
</figure>
</center>

<center>
<figure style="width:70%"> <img src="/Images/Study/cnn-translation/11.png" alt="Translation Equivariance"/>
</figure>
</center>

# References
1. [https://divsoni2012.medium.com/translation-invariance-in-convolutional-neural-networks-61d9b6fa03df](https://divsoni2012.medium.com/translation-invariance-in-convolutional-neural-networks-61d9b6fa03df)
2. [https://qr.ae/prNtyH](https://qr.ae/prNtyH)
3. [https://www.youtube.com/watch?v=rTawFwUvnLE&ab_channel=trwappers](https://www.youtube.com/watch?v=rTawFwUvnLE&ab_channel=trwappers)