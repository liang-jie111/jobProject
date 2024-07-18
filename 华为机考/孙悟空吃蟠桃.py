'''
题目描述
孙悟空爱吃蟠桃，有一天趁着蟠桃园守卫不在来偷吃。已知蟠桃园有 N 棵蟠桃树，每棵树上都桃子，守卫将在 H 小时后回来。
孙悟空可以决定他吃蟠桃的速度 K （个/每小时），每个小时选一棵桃树，并从树上吃掉 K 个，如果K大于该树上所有桃子个数，则全部吃掉，并且这一小时剩余的时间里不再吃桃。
孙悟空喜欢慢慢吃，但又想在守卫回来前吃完桃子。
请返回孙悟空可以在 H 小时内吃掉所有桃子的最小速度 K （K 为整数）。如果以任何速度都吃不完所有桃子，则返回 0。

输入描述
    第一行输入为 N个数字， N 表示桃树的数量，这 N 个数字表示每棵桃树上蟠桃的数量。
    第二行输入为一个数字，表示守卫离开的时间 H。
    其中数字通过空格分割， N、 H 为正整数，每棵树上都有蟠桃，且 0<N<10000, 0 < H < 10000。

输出描述
    输出吃掉所有蟠桃的最小速度 K，无解或输入异常时输出 0。

示例1
    输入：
    2 3 4 5
    4
    输出：
    5

示例2
    输入：
    2 3 4 5
    3
    输出：
    0

示例3
    输入：
    30 11 23 4 20
    6
    输出：
    23
'''
import math
from typing import List


def ok(peachs: List[int], speed: int, H: int) -> bool:
    """
    :param peachs: 每棵桃树上蟠桃的数量
    :param speed: 守卫每小时吃的桃子数量
    :param H: 守卫离开的时间
    :return:  每个小时只能选一棵桃树，能否在 H 小时内吃完所有的桃子
    """
    time = 0
    for cnt in peachs:
        # time += (cnt + speed - 1) // speed  # 向上取整
        time += math.ceil(cnt/speed)

    return time <= H


def solve(peachs: List[int], H: int) -> int:
    n = len(peachs)

    # 每个小时只能选一棵桃树，因此任何速度都吃不完所有桃子
    if n > H:
        return 0

    l, r = 0, max(peachs)
    while l + 1 < r:
        mid = (l + r) // 2
        if ok(peachs, mid, H):
            r = mid
        else:
            l = mid
    return r


if __name__ == '__main__':
    # 每棵桃树上蟠桃的数量
    # peachs = list(map(int, input().split()))
    peachs = [30, 11, 23, 4, 20]

    # 守卫离开的时间
    # H = int(input())
    H = 6

    print(solve(peachs, H))