"""
可以对一个字符串进行三种操作，插入一个字符，删除一个字符，替换一个字符
现在给你两个字符串s1和s2，请计算将s1转换成s2最少需要多少次操作

解决两个字符串的动态规划问题，一般都是用两个指针i,j分别指向两个字符串的末尾，然后一步步往前走，缩小问题的规模

base case:i走完s1或者j走完s2，可以直接返回另一个字符串剩余长度
"""
import numpy as np

'''
方法一、暴力动态规划
'''
def minDistance(s1, s2):
    def dp(i, j):
        '''
            i和j是状态。
            base case是某一个字符串探索完了
        '''
        if i < 0:
            return j+1
        if j < 0:
            return i+1

        '''
        选择：对于每对字符s1[i]和s2[j]，有四种操作：跳过（相等的话）、插入、删除、替换
        '''
        if s1[i] == s2[j]:
            return dp(i-1, j-1)
        else:
            return min(
                dp(i, j-1) + 1, #插入  直接在s1中插入一个和s2[j]一样的字符，那么s2[j]就被匹配了，前移j，继续和i对比，别忘了操作数+1
                dp(i-1, j) + 1, #删除  直接把s1[i]这个字符删掉，前移i，继续和j对比，操作数+1
                dp(i-1, j-1) + 1 #替换  直接把s1[i]替换成s2[j]，这样他俩就匹配了，同时前移i和j继续对比，操作数+1
            )
    return dp(len(s1)-1, len(s2)-1)

'''
优化一、加备忘录
'''
def minDistance2(s1, s2):
    memo = dict()
    def dp(i, j):
        # 先检查备忘录
        if (i, j) in memo:
            return memo[(i, j)]

        '''
            i和j是状态。
            base case是某一个字符串探索完了
        '''
        if i < 0:
            return j+1
        if j < 0:
            return i+1

        '''
        选择：对于每对字符s1[i]和s2[j]，有四种操作：跳过（相等的话）、插入、删除、替换
        '''
        if s1[i] == s2[j]:
            memo[(i, j)] = dp(i-1, j-1)
        else:
            memo[(i, j)] = min(
                dp(i, j-1) + 1, #插入  直接在s1中插入一个和s2[j]一样的字符，那么s2[j]就被匹配了，前移j，继续和i对比，别忘了操作数+1
                dp(i-1, j) + 1, #删除  直接把s1[i]这个字符删掉，前移i，继续和j对比，操作数+1
                dp(i-1, j-1) + 1 #替换  直接把s1[i]替换成s2[j]，这样他俩就匹配了，同时前移i和j继续对比，操作数+1
            )
        return memo[(i, j)]

    return dp(len(s1)-1, len(s2)-1)

'''
优化二、DP Table动态规划
'''
def minDistance3(s1, s2):
    m, n = len(s1), len(s2)
    dp = np.array([[0]*(n+1)]*(m+1))

    for i in range(1, m+1):
        dp[i][0] = i
    for j in range(1, n+1):
        dp[0][j] = j

    print(dp)

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j]+1, #删除
                    dp[i][j-1]+1, #插入
                    dp[i-1][j-1]+1 #替换
                )

    return dp[m][n]


'''
扩展：输出具体的操作步骤
'''
class Node:
    def __init__(self, val, choice):
        """Constructor of the class."""
        self.val = val
        self.choice = choice
        '''
        0:什么都不做
        1：插入
        2：删除
        3：替换
        '''
def min_Node(node1, node2, node3):
    res = Node(node1.val, 2)
    if res.val > node2.val:
        res.val = node2.val
        res.choice = 1

    if res.val > node3.val:
        res.val = node3.val
        res.choice = 3
    return res

def printRes(dp, s1, s2):
    rows, cols = np.shape(dp)[0], np.shape(dp)[1]
    i, j = rows - 1, cols - 1
    print(f"Change s1={s1} to s2={s2}:")
    while i != 0 and j != 0:
        c1, c2 = s1[i - 1], s2[j - 1]
        choice = dp[i][j].choice
        print(f"s1[{i-1}]:")
        if choice==0:
            print(f"skip '{c1}'")
            i-=1
            j-=1
        elif choice==1:
            print(f"insert '{c2}'")
            j-=1
        elif choice==2:
            print(f"delete '{c1}'")
            i-=1
        # if choice==3:
        else:
            print(f"replace '{c1}' with '{c2}'")
            i-=1
            j-=1

    while i>0:
        print(f"delete s1[{i-1}]={s1[i-1]}")
        i-=1
    while j>0:
        print(f"insert s2[{j - 1}]={s2[j - 1]}")
        j -= 1

def minDistance4(s1, s2):
    m, n = len(s1), len(s2)
    dp = np.array([[Node(0, 0)]*(n+1)]*(m+1))

    for i in range(1, m+1):
        # s2空了，s1转换成s2只需要删除字符
        dp[i][0] = Node(i, 2)

    for j in range(1, n+1):
        # s1空了，s1转换成s2只需要插入字符
        dp[0][j] = Node(j, 1)


    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = Node(dp[i-1][j-1].val, 0)
            else:
                dp[i][j] = min_Node(
                    dp[i-1][j], #删除
                    dp[i][j-1], #插入
                    dp[i-1][j-1] #替换
                )
                dp[i][j].val += 1

    printRes(dp, s1, s2)

    return dp[m][n].val

s1 = 'rad'
s2 = 'apple'
print(minDistance(s1, s2))
print(minDistance2(s1, s2))
print(minDistance3(s1, s2))
print(minDistance4(s1, s2))