'''
一只贪吃的猴子，来到一个果园，发现许多串香蕉排成一行，每串香蕉上有若干根香蕉。每串香蕉的根数由数组numbers给出。猴子获取香蕉，每次都只能从行的开头或者末尾获取，并且只能获取N次，求猴子最多能获取多少根香蕉。

输入描述
    第一行为数组numbers的长度
    第二行为数组numbers的值每个数字通过空格分开
    第三行输入为N，表示获取的次数
输出描述
    按照题目要求能获取的最大数值

输入
4
4 2 2 3
2
输出
7
说明
第一次获取香蕉为行的开头，第二次获取为行的末尾，因此最终根数为4+3 =7
'''
from collections import deque
n = 4
lst = deque([4, 2,4])
times = 2

state = []
max_nums = 0

def backtrak():
    global max_nums
    if len(state) == times:
        max_nums = max(sum(state), max_nums)
        return
    num = lst.popleft()
    state.append(num)
    backtrak()
    state.pop()
    lst.appendleft(num)

    num = lst.pop()
    state.append(num)
    backtrak()
    state.pop()
    lst.append(num)


backtrak()
print(max_nums)





n = 4
numbers = [4,2,2,3]
N = 2

tot = sum(numbers[i] for i in range(N))
rs = tot

r = n - 1
for l in range(N-1, -1, -1):
    tot += numbers[r] - numbers[l]
    rs = max(rs, tot)
    r -= 1

print(rs)