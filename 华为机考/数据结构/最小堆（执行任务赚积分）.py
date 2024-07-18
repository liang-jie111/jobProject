'''
题目描述
    现有 N 个任务需要处理，同一时间只能处理一个任务，处理每个任务所需要的时间固定为 1。
    每个任务都有最晚处理时间限制和积分值，在最晚处理时间点之前处理完成任务才可获得对应的积分奖励。
    可用于处理任务的时间有限，请问在有限的时间内，可获得的最多积分。

输入描述
    第一行为一个数 N ，表示有 N 个任务（1 ≤ N ≤ 100 ）
    第二行为一个数 T ，表示可用于处理任务的时间。（1 ≤ T ≤ 100）
    接下来 N 行，每行两个空格分隔的整数（SLA 和 和 V ），SLA 表示任务的最晚处理时间，V 表示任务对应的积分。
    1≤SLA≤100 , 0 ≤ V ≤ 100000

输出描述
    可获得的最多积分

示例2
输入：
4
3
1 2
1 3
1 4
3 5
输出：
9

说明：
第 1 个时间单位内，处理任务3，获得 4 个积分
第 2 个时间单位内，处理任务 4，获得 5 个积分
第 3 个时间单位内，无任务可处理。
共获得 9 个积分
'''
import heapq
'''
heapq-堆排序算法：heapq实现了一个适合与Python的列表一起使用的最小堆排序算法。
    heappush(heap,item)建立大、小根堆
    heapq.heappush()是往堆中添加新值，此时自动建立了小根堆.但heapq里面没有直接提供建立大根堆的方法，可以采取如下方法：每次push时给元素加一个负号（即取相反数），此时最小值变最大值，反之亦然，那么实际上的最大值就可以处于堆顶了，返回时再取负即可。
    
    heapify(heap)建立大、小根堆
    heapq.heapfiy()是以线性时间将一个列表转化为小根堆（复杂度O（logn））：
    a = [1, 5, 20, 18, 10, 200]
    heapq.heapify(a)
    
    heappop(heap)
    使用heappop()弹出并返回堆中的最小项，保持堆不变。如果堆是空的，则引发IndexError。
    heapq.heappop()是从堆中弹出并返回最小的值
    
    
'''



def fun():
    # n = int(input())
    # t = int(input())
    n = 4
    t = 3

    # tasks = [tuple(map(int, input().split())) for _ in range(n)]
    tasks = [(1,2), (1,3), (1,4), (3,5)]
    # 对任务进行排序（时间升序）
    tasks.sort()

    h = []
    for last, v in tasks:
        heapq.heappush(h, v)

        # last 时间最多只能完成 last 个任务，任务无法在 last 时间内完成，则移除积分值最小值（最小堆）的任务
        # 可用于处理任务的时间 t 最多只能完成 t 个任务
        if last < len(h) or len(h) > t:
            heapq.heappop(h)

    print(sum(h))


if __name__ == "__main__":
    fun()
