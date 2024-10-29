---
title: "스마트 포인터 (Smart Pointer)"
categories:
- cpp
tags: [Memory]
toc: false
toc_sticky: true
toc_label: "On this page"
published: true
use_math: true
---

# Garbage Collection
프로그래밍을 하다보면, 메모리를 동적할당하는 경우가 아주 빈번합니다. 동적할당된 메모리를 컴퓨터가 더이상 사용하지 않을 때, 이 메모리를 해제하여 사용할 수 있는 공간을 확보하는 것이 가비지 컬렉션 기법입니다. Go, Java, Javascript, python 등의 언어는 이러한 기능을 지원하는데, C/C++의 new를 통한 동적할당은 이를 **지원하지 않습니다**. 즉, `new`를 통해 할당해준 메모리는 반드시 사용자가 `delete`로 삭제해주어야만 메모리 누수가 발생하지 않게 됩니다.

C++에서도 GC를 위한 기능이 있는데, C++11 이전에는 `std::auto_ptr`을 통해 구현되었고, 그 이후로는 `unique_ptr`, `shared_ptr`, `weak_ptr`의 세가지를 통해 활용되고 있습니다. 불필요해진 auto_ptr는 C++17에서 제거되었습니다.

# unique_ptr


