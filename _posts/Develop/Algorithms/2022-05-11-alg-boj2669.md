---
title: "BOJ 2669 - 직사각형 네개의 합집합의 면적 구하기"
excerpt: "BaekJoon Online Judge 2669"
categories:
  - Algorithms
tags: [Python, Algorithms, BOJ]
toc: false
toc_sticky: true
toc_label: "On this page"
published: true
use_math: true

date: 2022-05-11
last_modified_at: 2022-05-11
---

## [2669 - 직사각형 네개의 합집합의 면적 구하기](https://www.acmicpc.net/problem/2669)
직사각형 네개의 정보를 받아와서 면적을 구하는 문제이다.
100x100으로 전체 도메인의 크기가 정해져있고, 정수형으로 한정되기 때문에 단순하게 100x100의 2d array로 만들어서
사각형이 있는 영역을 1로 만들어주고, 나중에 1의 개수를 카운트해주면 간단하다.

<center>
<figure style="width: 60%"> <img src="/Images/Algorithms/boj2699-grid.jpg" alt="BOJ 2699"/>
<figcaption>면적 예시</figcaption>
</figure>
</center>

```python
grid = [[0]*100 for _ in range(101)]
for _ in range(4):
    minx, miny, maxx, maxy = map(int,input().split())
    for x in range(minx, maxx):
        for y in range(miny, maxy):
            grid[y][x] = 1
area = 0
for ax in grid:
    area += ax.count(1)
print(area)
```