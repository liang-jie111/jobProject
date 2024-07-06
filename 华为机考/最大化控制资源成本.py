'''
题目描述
公司创新实验室正在研究如何最小化资源成本，最大化资源利用率，请你设计算法帮他们解决一个任务分布问题:有taskNum项任务，每人任务有开始时间(startTime) ，结更时间(endTme)
并行度(paralelism) 三个属性，并行度是指这个任务运行时将会占用的服务器数量，一个服务器在每个时刻可以被任意任务使用但最多被一个任务占用，任务运行完成立即释放(结束时刻不占用)。
任务分布问题是指给定一批任务，让这批任务由同一批服务器承载运行，请你计算完成这批任务分布最少需要多少服务器，从而最大最大化控制资源成本。
输入描述
    第一行输入为taskNum，表示有taskNum项任务 接下来taskNum行，每行三个整数，表示每个任务的开始时间(startTime ) ，结束时间 (endTime ) ，并行度 (parallelism)
输出描述
    一个整数，表示最少需要的服务器数量

输入
2
3 9 2
4 7 3
输出
5
说明
共两个任务，第一个任务在时间区间[3，9]运行，占用2个服务器，第二个任务在时间区间[4，7] 运行，占用3个服务器，需要最多服务器的时间区间为[4，7] ，需要5个服务器

差分数组 题
解题思路：
    使用差分数组的思想，创建一个数组 d 来记录每个时间点的服务器占用情况。遍历每个任务，将其开始时间和结束时间对应的数组元素进行累加和累减，表示服务器的占用情况。在遍历的过程中，记录累加和的最大值，即为需要的最大服务器数量。
'''
task_num = int(input())
# 差分数组
N = 50000 + 5
d = [0] * N

for _ in range(task_num):
    start_time, end_time, parallelism = map(int, input().split())
    d[start_time] += parallelism
    d[end_time] -= parallelism

result = 0
sum_value = 0
for i in range(N):
    # sum_value 表示在 i 时间，执行所有任务所需服务器数量
    sum_value += d[i]
    result = max(result, sum_value)

print(result)
