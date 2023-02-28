---
title: "Struct Memory Alignment, Padding/Packing"
categories:
- cpp
tags: [Memory]
toc: false
toc_sticky: true
toc_label: "On this page"
published: true
use_math: true

---

# Struct Memory Alignment
C++에서 구조체/클래스는 컴파일러에 의해 메모리가 정렬됩니다. 아래와 같은 자료형을 정의했을 때, 실제로 우리가 사용하는 변수의 메모리를 계산하면 총 13바이트임에도 내부적으로 메모리에 빈 값이 들어가서 규칙적으로 정렬되어 더 많은 메모리를 차지합니다.

```cpp
struct Case0
{
    bool flag;  // 1-byte
    // pad 3-bytes
    int number;  // 4-bytes
    double real;  // 8-bytes
};
```

이 경우에는 1바이트인 boolean flag 변수 뒤에 3바이트만큼의 패딩이 들어가게 되는 것입니다. 그래서 구조체의 메모리 크기를 출력하기 위해 `sizeof(Case0)`을 하면 계산했을 때의 13바이트가 아닌 16바이트가 출력됩니다. 
이번엔 아래와 같은 경우를 생각해 보겠습니다.

```cpp
struct Case1
{
    bool flag;  // 1-byte
    // pad 7-bytes;
    double real;  // 8-bytes
    int number;  // 4-bytes
    // pad 4-bytes
};
```

저런. 단순히 변수 순서만 바꾸었는데 패딩이 더 늘어나서 기존보다 더 많은 메모리를 필요로 하게 되었습니다. 이를 그림으로 나타내면 다음과 같습니다.

<center>
<figure style="width: 50%"> <img src="/Images/devstudy/struct-memory/img1.jpg" alt="C++ Struct Memory Padding"/>
<figcaption> Memory Padding </figcaption>
</figure>
</center>

이렇게 메모리를 더 쓰는 이유는 보다 효율적인 메모리 접근을 위해서입니다. CPU가 한번에 처리할 수 있는 데이터의 크기를 `WORD`로 정의하는데, 요즘의 컴퓨터는 대부분 64비트 운영체제로, 워드의 크기가 8바이트입니다. 각 타입의 변수들은 지정된 메모리 주소의 배수에 위치하게 됩니다. 그렇지 않을 경우, 다음 그림과 같이, 한 번에 처리할 수 있는 연산이 두 번으로 늘어나게 됩니다. 이러한 이유로 구조체의 멤버변수들은 패딩을 통해 정해진 메모리 공간에 위치하게 됩니다.

<center>
<figure style="width: 50%"> <img src="/Images/devstudy/struct-memory/img3.jpg" alt=""/>
<figcaption> 메모리 접근 </figcaption>
</figure>
</center>

# Packing
하지만 컴퓨터 자원의 한계를 포함한 여러 문제로 인해 이러한 패딩을 일부러 없애야 하는 경우도 있을 수 있습니다. 패딩을 없애는 것을 `패킹(Packing)`이라고 하며, 대부분의 컴파일러들이 패킹을 지원합니다. [링크](https://gcc.gnu.org/onlinedocs/gcc-4.4.7/gcc/Structure_002dPacking-Pragmas.html)에도 간략하게 나와있으니 필요시 참고하시면 좋을 것 같습니다.

```cpp
#pragma pack(push, 1)
struct Case2
{
    bool flag;  // 1-byte
    double real;  // 8-bytes
    int number;  // 4-bytes
};
#pragma pack(pop)
```

`#pragma` 지시문을 통해 패딩을 얼마만큼 할지 지정할 수 있으며, 1로 설정하면 패딩을 하지 않는 것을 의미합니다. 패딩을 하지 않으면 아래의 그림과 같이 압축된 자료형이 되고, 메모리를 측정하면 계산했던 13바이트가 나오게 됩니다. 대신 아까 살펴본 것과 같이, 메모리 주소가 일치하지 않으므로 성능저하가 발생할 가능성이 있습니다.

<center>
<figure style="width: 50%"> <img src="/Images/devstudy/struct-memory/img2.jpg" alt="C++ Struct Memory Padding"/>
<figcaption> Memory Packing </figcaption>
</figure>
</center>

# alignas, alignof
앞서 살펴본 것과는 별개로, `alignas` 지시어를 통해 메모리 정렬을 임의로 설정할 수 있습니다. 이 기능은 C++11부터 지원합니다.
- 구조체/클래스의 선언/정의부
- 비트필드 클래스 멤버가 아닌 선언부
- 다음에 속하지 않는 변수의 선언부
  - 함수 파라미터
  - catch 절의 예외변수

```cpp
struct alignas(size) S { };
``` 

사용법은 위와 같습니다. 이 때, alignas(3)과 같이, 2진법으로 정렬되지 않는 경우는 무시되며(invalid non-zero alignments), 멤버들 중 가장 큰 크기보다 정렬 값이 작을 경우도 무시됩니다. 예를 들어, 8바이트 크기의 double 타입 변수가 있는 경우에 alignas(4)와 같은 경우는 무시됩니다.

```cpp
struct Case3
{
    static void PrintSize()
    {
        cout << "[Case 3]" << endl;
        cout << "Size: " << sizeof(Case3) << " bytes" << endl;
        cout << "Align: " << alignof(Case3) << " bytes" << endl;
        cout << endl;
    }

    bool flag;  // 1-byte
    float real;  // 4-bytes
    int number;  // 4-bytes
};

struct alignas(double) Case4
{
    static void PrintSize()
    {
        cout << "[Case 4]" << endl;
        cout << "Size: " << sizeof(Case4) << " bytes" << endl;
        cout << "Align: " << alignof(Case4) << " bytes" << endl;
        cout << endl;
    }

    bool flag;  // 1-byte
    float real;  // 4-bytes
    int number;  // 4-bytes
};
```

위 두 구조체의 크기를 출력해보면 아래와 같이 나옵니다. alignof는 해당 객체의 메모리 정렬 크기를 나타냅니다.

```bash
[Case 3]
Size: 12 bytes
Align: 4 bytes

[Case 4]
Size: 16 bytes
Align: 8 bytes
```