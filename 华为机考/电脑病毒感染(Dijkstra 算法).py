'''
题目描述
一个局域网只内有很多台电脑，分别标注为 1 ~ N 的数字。相连接的电脑距离不一样，所以感染时间不一样，感染时间用t 表示。
其中网络内一台电脑被病毒感染，求其感染网络内所有的电脑最少需要多长时间。如果最后有电脑不会感染，则返回-1。
给定一个数组 times 表示一台电脑把相邻电脑感染所用的时间: path[i] = {i, j, t} 表示: 电脑i 上的病毒感染 j，需要时间 t 。

输入描述
    第一行输入一个整数N，表示局域网内电脑个数 N，1<= N<= 200 ；
    第二行输入一个整数M, 表示有 M 条网络连接；
    接下来M行, 每行输入为 i,j,t 。表示电脑 i 感染电脑 j 需要时间t。(1 <= i, j <= N)
    最后一行为病毒所在的电脑编号。
输出描述
    输出最少需要多少时间才能感染全部电脑，如果不存在输出 -1

题解
    经典的最短路径问题，使用 Dijkstra 算法求解。

代码思路：
    1、 图的构建：使用 g 表示图，其中键为电脑编号，值为与该电脑相邻的电脑以及感染所需的时间。
    2、优先队列：使用 heap 作为优先队列，队列中存储的是数组 {时间, 电脑编号}。
    3、Dijkstra 算法：开始时，将病毒所在电脑加入队列，并设置时间为0。然后，不断从队列中取出时间最小的电脑，将其相邻的未感染电脑加入队列，并更新感染时间。重复这个过程直到所有电脑都被感染或队列为空。
    4、输出结果：根据感染情况输出结果。
'''
from collections import defaultdict
from heapq import heappop, heappush

N = int(input())
M = int(input())

# 构建图
g = defaultdict(set)
for _ in range(M):
    i, j, t = map(int, input().split())
    g[i].add((j, t))

# 病毒所在的编号, 优先级队列，已经感染的电脑集合
sid, heap, vis = int(input()), [], set()

heappush(heap, (0, sid))
while heap:
    time, id = heappop(heap)
    vis.add(id)

    if N == len(vis):  # 感染了所有电脑
        print(time)
        break

    for next_id, cost_time in g[id]:
        if next_id not in vis:  # 此时 next_id 电脑还没有被感染
            heappush(heap, (time + cost_time, next_id))

if N != len(vis):
    print(-1)