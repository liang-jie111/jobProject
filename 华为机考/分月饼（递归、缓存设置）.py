'''
题目描述
中秋节，公司分月饼，m个员工，买了n个月饼，m<=n，每个员工至少分1个月饼，但可以分多个，单人分到最多月饼的个数是Max1，
单人分到第二多月饼个数是Max2，Max1-Max2<=3，
单人分到第n-1多月饼个数是Max(n-1)，单人分到第n多月饼个数是Max(n)，Max(n-1)-Max(n)<=3问有多少种分月饼的方法?

输入描述
每一行输入m n，表示m个员工，n个月饼，m<=n

输出描述
输出有多少种月饼分法

输入：
3 12

输出：
6

说明：
满足要求的有6种分法:
12=1+1+10(Max1=10,Max2=1，不满足要求)
12=1+2+9(Max1=9,Max2=2,不满足要求)
12=1+3+8(Max1=8,Max2=3 不满足要求)
12=1+4+7(Max1=7,Max2=4,Max3=1, 满足要求)
12=1+5+6(Max1=6,Max2=5,Max3=1，不满足要求)
12=2+2+8(Max1=8,Max2=2,不满足要求)
12=2+3+7(Max1=7,Max2=3,不满足要求)
12=2+4+6(Max1=6,Max2=4,Max3=2，满足要求)
12=2+5+5(Max1=5,Max2=2，满足要求)
12=3+3+6(Max1=6,Max2=3,满足要求)
12=3+4+5(Max1=5,Max2=4,Max3=3，满足要求)
12=4+4+4(Max1=4,满足要求)

题解
    这个问题可以通过递归的方式解决。
    首先，每个员工至少分到一个月饼，因此可以将每个员工分到一个月饼后，剩余的月饼数量为n - m。然后，可以枚举分配多余月饼的情况，并递归计算方案数。
    因题目 （1,2,3）（1,3,2）（2,1,3）（2,3,1） 是一同一种分配月饼的方法，因此为了方便我们枚举分配情况，
    我考虑使用非递减的方式来分配月饼（即第 i + 1个人分配的月饼数 >= 第 i 个人分配的月饼数）。
    递归的方法定义就是：
    // 有m个人n个月饼，每人最少分配 start 个月饼的分配方案数 int assign(int m, int n, int start)
'''
from functools import cache
import sys
sys.setrecursionlimit(10000)  # 没有这个只能 96%

@cache
def assign(m: int, n: int, start: int) -> int:
    """ 有m个人n个月饼，每人最少分配 start 个月饼的分配方案数"""
    if start * m > n or m < 0 or n < 0:  # 不满足分配条件
        return 0
    if m == 0 and n == 0:  # 分配完成
        return 1

    cnt = 0
    up = min(start + 3, n)  # 下次分配最多可以分配的个数
    for k in range(start, up + 1):
        cnt += assign(m - 1, n - k, k)
    return cnt


m, n = map(int, input().split())
# 先每人分配一个月饼先，因此剩余的月饼数是  n- m
# 然后再枚举多余的元素再分的情况
tot = sum([assign(m - 1, n - m - i, i) for i in range(n - m + 1)])
print(tot)