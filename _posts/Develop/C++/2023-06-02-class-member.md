---
title: "C++ 에서 클래스 멤버를 모두 순회하기"
categories:
- cpp
tags: [Memory, Class]
toc: false
toc_sticky: true
toc_label: "On this page"
published: false
use_math: true

---

CUDA 로 프로그래밍을 하던 와중, 보다 효율적으로 host/device 메모리를 관리하기 위한 객체를 정의했다.

```cpp
class HostClass {
public:
    std::vector<int> a;
    std::vector<double> b;
    std::vector<float> c;
};

class DeviceClass {
public:
    DeviceClass() {}
    ~DeviceClass() {
        if (a) {
            cudaFree(a);
            a = nullptr;
        }
        if (b) {
            cudaFree(b);
            b = nullptr;
        }
        if (c) {
            cudaFree(c);
            c = nullptr;
        }
    }

public:
    int* a = nullptr;
    double* b = nullptr;
    float* c = nullptr;
};
```