'''
最长递增子序列
'''
import numpy as np


def lengthOfList(nums):
    n = len(nums)
    dp = [1] * n

    for i in range(n):
        for j in range(0, i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    print(dp)
    res = 0
    for i in range(n):
        res = max(res, dp[i])
    return res


'''效率进阶'''
import bisect  # 导入查找模块
def get_max_sub(arr):  # 定义获取最长子序列函数
    res = [arr[0]]  # 将传入的列表第一个参数放入res
    dp = [1] * len(arr)  # 定义一个长度为输入列表长度的列表，元素为1.
    for i in range(1, len(arr)):  # 计算以arr[i]结尾的最长上升子序列长度
        if arr[i] > res[-1]:  # 如果arr[i]大于最后一个元素，插入
            res.append(arr[i])
            dp[i] = len(res)
        else:  # 如果arr[i]小于最后一个元素，找到res中比他大的元素的位置，并将该元素替换为arr[i]
            index = bisect.bisect_left(res, arr[i])
            res[index] = arr[i]
            dp[i] = index + 1
    return dp


# nums = [1,2,3,1,2,3,4,1,3,7]
nums = [10, 3, 9, 11, 10, 109, 2]
print(lengthOfList(nums))
print(get_max_sub(nums))

'''
二维递增子序列：信封嵌套问题
实际是最长递增子序列问题上升到二维，解法需要先按照特定的规则排序，之后转换为一个一维的最长递增子序列问题，最后用二分搜索技巧解决
'''


def maxEnvelopes(envelopes):
    '''
    先对宽度w升序排序，w相同，高度h降序排序。排好序后，对所有h组成的序列进行一维的最长自增子序列计算
    '''
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    print(envelopes)
    height = [row[1] for row in envelopes]

    return lengthOfList(height)


envelopes = [[5, 4], [6, 5], [6, 6], [2, 3], [1, 1]]
print(maxEnvelopes(envelopes))

'''
装箱子问题
'''

'''
最大子数组和问题
'''
def maxSubArray(nums):
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    for i in range(n):
        dp[i] = max(nums[i], nums[i] + dp[i - 1])
    return max(dp)


def maxSubArray2(nums):
    '''
    可以发现dp[i]只和dp[i-1]有关，可以进行状态压缩，降低空间复杂度
    '''
    n = len(nums)
    dp0, dp1, res = nums[0], 0, nums[0]
    for i in range(n):
        dp1 = max(nums[i], nums[i] + dp0)
        dp0 = dp1
        res = max(res, dp1)
    return res


nums = [-3, 1, 3, -1, 2, -4, 2]
print(maxSubArray(nums))
print(maxSubArray2(nums))

'''
最长公共子序列

第一步：明确dp数组的定义。dp[i][j]：对于s1[0...i-1]和s2[0...j-1]，它们的LCS长度是dp[i][j]
第二步：定义base case。专门让索引为0的行和列表示空串，dp[0][...]和dp[...][0]都应该初始化为0
第三步：找状态转移方程。如果s1[i]==s2[j]，说明这个字符一定在lcs中，则dp[i-1][j-1]+1就是dp[i][j];否则，说明s1[i]和s2[j]至少有一个不在lcs中，取最大即可
'''


def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1)] * (m + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m - 1][n - 1]


s1 = 'abcde'
s2 = 'baczex'
print(lcs(s1, s2))

'''
最长连续公共子序列
'''
def maxSub(a_str, b_str):
    import numpy as np
    n, m = len(a_str), len(b_str)
    dp = np.array([[0] * n] * m)

    for i in range(0, n):
        if b_str[0] == a_str[i]:
            dp[0][i] = 1
    for j in range(0, m):
        if b_str[j] == a_str[0]:
            dp[j][0] = 1

    for i in range(1, m):
        for j in range(1, n):
            if a_str[j] == b_str[i]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            # else:
            #   dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    print(dp)
    # return dp[m - 1][n - 1]
    return max([max(row) for row in dp])


a_str = "abdef"
b_str = "bdabdef"
print(maxSub(a_str, b_str))


'''
给你两棵二叉树的根节点 p 和 q ，编写一个函数来检验这两棵树是否相同。
如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
'''


class Solution:
    def isSameTree(self, p, q) -> bool:
        if p is None or q is None:
            return p is q  # 必须都是 None
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)


'''
给你一个二叉树的根节点 root ， 检查它是否轴对称。
'''


class Solution:
    def isSymmetric(self, root) -> bool:

        def dfs(left, right):

            if not left and not right:
                return True
            elif not left and right:
                return False
            elif not right and left:
                return False

            if left.val != right.val:
                return False
            else:
                return dfs(left.left, right.right) and dfs(left.right, right.left)

        return dfs(root.left, root.right)


'''
跳跃游戏
给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。
判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false

思路一  贪心：
    尽可能到达最远位置（贪心）。
    如果能到达某个位置，那一定能到达它前面的所有位置。
方法：
    初始化最远位置为 0，然后遍历数组，如果当前位置能到达，并且当前位置+跳数>最远位置，就更新最远位置。最后比较最远位置和数组长度。

思路二  动态规划+回溯：
定义 dp[i] 表示从 0 出发，经过 j<=i，可以跳出的最远距离。

初始化: dp[0]=nums[0]
迭代: 如果能通过前 i−1 个位置到达 i，即 dp[i−1]>=i, 那么 dp[i]=max(dp[i−1],i+nums[i])，否则 dp[i]=dp[i−1]
'''


def canJump(self, nums):
    max_i = 0  # 初始化当前能到达最远的位置
    for i, jump in enumerate(nums):  # i为当前位置，jump是当前位置的跳数
        if max_i >= i and i + jump > max_i:  # 如果当前位置能到达，并且当前位置+跳数>最远位置
            max_i = i + jump  # 更新最远能到达位置
    return max_i >= i


def canJump(nums):
    dp = [0 for _ in range(len(nums))]
    dp[0] = nums[0]
    for i in range(1, len(nums)):
        if dp[i - 1] >= i:
            dp[i] = max(dp[i - 1], i + nums[i])
        else:
            dp[i] = dp[i - 1]
    return dp[-1] >= len(nums) - 1


'''
跳跃游戏2
给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。
每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j] 处:
0 <= j <= nums[i]
i + j < n
返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。
'''


# 思路 1：动态规划+回溯（超时）代码
class Solution:
    def jump(self, nums: list[int]) -> int:
        size = len(nums)
        dp = [float("inf") for _ in range(size)]
        dp[0] = 0

        for i in range(1, size):
            for j in range(i):
                if j + nums[j] >= i:
                    dp[i] = min(dp[i], dp[j] + 1)

        return dp[size - 1]


# 思路 2：动态规划+回溯 + 贪心
class Solution:
    def jump(self, nums: list[int]) -> int:
        size = len(nums)
        dp = [float("inf") for _ in range(size)]
        dp[0] = 0

        j = 0
        for i in range(1, size):
            while j + nums[j] < i:
                j += 1
            dp[i] = dp[j] + 1

        return dp[size - 1]


# 思路 3：贪心算法
class Solution:
    def jump(self, nums: list[int]) -> int:
        end, max_pos = 0, 0
        steps = 0
        for i in range(len(nums) - 1):
            max_pos = max(max_pos, nums[i] + i)
            if i == end:
                end = max_pos
                steps += 1
        return steps


'''
爬楼梯
'''


def climbStairs(n):
    if n <= 3:
        return n
    a = 1
    b = 2
    c = 0
    for i in range(3, n + 1):
        c = a + b
        a = b
        b = c
    return c


def searchMatrix(matrix, target):
    m, n = len(matrix), len(matrix[0])
    print(m, n)

    if target < matrix[0][0] or target > matrix[m - 1][n - 1]:
        return False

    # 第一次二分法确定在哪一行
    l1, r1 = 0, m - 1
    print("l1=", l1, "   r1=", r1)

    while l1 < r1:
        print("1")
        mid1 = (l1 + r1) >> 1
        if target == matrix[mid1][0] or target == matrix[l1][0] or target == matrix[r1][0]:
            print("12")
            return True
        elif target > matrix[mid1][0]:
            print("13")
            l1 = mid1 + 1
        else:
            print("14")
            r1 = mid1 - 1
    print(l1, r1)
    if target < matrix[l1][0]:
        curr_row = l1 - 1
    else:
        curr_row = l1
    print(curr_row)

    l2, r2 = 0, n - 1
    while l2 <= r2:
        print("2")
        mid2 = (l2 + r2) >> 1
        # if target == matrix[curr_row][mid2] or target == matrix[curr_row][l2] or target == matrix[curr_row][r2]:
        if target == matrix[curr_row][mid2]:
            print('21')
            return True
        elif target > matrix[curr_row][mid2]:
            print('22')
            l2 = mid2 + 1
        else:
            print('23')
            r2 = mid2 - 1
    print('24')
    return False


# searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 60)
searchMatrix([[1, 1], [2, 2]], 2)


'''
一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：
'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。
'''


class Solution:
    def numDecodings(self, s: str) -> int:
        L = len(s)
        dp = [0] * (L + 1)
        if s[0] == '0':
            return 0
        if L == 1:
            return 1

        dp[0], dp[1] = 1, 1

        for i in range(1, L):
            # 第i位置的字符要么独立加进来,s[i]!='0'时，dp[i]=dp[i-1]+1
            # 要么与i-1位的字符结合,如果"00"<s[i-1:i+1]<"27".dp[i] = dp[i-2]+1
            if s[i] != '0':
                if "09" < s[i - 1:i + 1] < "27":  # 可以结合也可以单独
                    dp[i + 1] = dp[i] + dp[i - 1]
                else:  # 不能结合，只能单独
                    dp[i + 1] = dp[i]
            else:  # 只能与前一个字符结合
                if "09" < s[i - 1:i + 1] < "27":
                    dp[i + 1] = dp[i - 1]
                else:
                    return 0
        print(dp)
        return dp[-1]


'''
给定三个字符串 s1、s2、s3，请你帮忙验证 s3 是否是由 s1 和 s2 交错 组成的。
两个字符串 s 和 t 交错 的定义与过程如下，其中每个字符串都会被分割成若干 非空子字符串：
s = s1 + s2 + ... + sn
t = t1 + t2 + ... + tm
|n - m| <= 1
交错 是 s1 + t1 + s2 + t2 + s3 + t3 + ... 或者 t1 + s1 + t2 + s2 + t3 + s3 + ...
注意：a + b 意味着字符串 a 和 b 连接。
'''


class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        len1 = len(s1)
        len2 = len(s2)
        len3 = len(s3)
        if (len1 + len2 != len3):
            return False
        dp = [[False] * (len2 + 1) for i in range(len1 + 1)]  # 初始化False矩阵
        dp[0][0] = True
        for i in range(1, len1 + 1):
            '''
            初始化第一列。
            dp[i][0]=dp[i−1][0] and s1[i−1]==s3[i−1]。表示 s1的前i位是否能构成s3的前i位。
            因此需要满足的条件为，前i−1位可以构成 s3的前i−1位且s1的第i位 s1[i−1]等于s3的第i位 s3[i−1]
            '''
            dp[i][0] = (dp[i - 1][0] and s1[i - 1] == s3[i - 1])
        for i in range(1, len2 + 1):
            '''
            初始化第一行。
            dp[0][i]=dp[0][i−1] and s2[i−1]==s3[i−1]。表示s2的前i位是否能构成s3的前i位。
            因此需要满足的条件为，前i−1位可以构成s3的前i−1位且s2的第i位 s2[i−1]等于s3的第i位 s3[i−1]
            '''
            dp[0][i] = (dp[0][i - 1] and s2[i - 1] == s3[i - 1])
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                '''
                s1前i位和s2的前j位能否组成s3的前i+j位取决于两种情况：
                    1.s1的前i−1个字符和s2的前j个字符能否构成s3的前i+j−1位，且s1的第i位 s1[i−1]是否等于s3的第i+j位 s3[i+j−1]
                    2.s1的前i个字符和s2的前j−1个字符能否构成s3的前i+j−1位，且s2的第j位 s2[j−1]是否等于s3的第i+j位 s3[i+j−1]

                '''
                dp[i][j] = (dp[i][j - 1] and s2[j - 1] == s3[i + j - 1]) or (
                            dp[i - 1][j] and s1[i - 1] == s3[i + j - 1])
        return dp[-1][-1]


m, n = list(map(int, input().split()))



'''
分苹果
把m个同样的苹果放在n个同样的盘子里，允许有的盘子空着不放，问共有多少种不同的分法？注意：如果有7个苹果和3个盘子，（5，1，1）和（1，5，1）被视为是同一种分法。
数据范围：0≤𝑚≤10   1≤𝑛≤10
'''
############ 动态规划
# 设dp[i][j]表示有i个苹果，j个盘子的情况下，有多少种可能，我们可以知道总共有至多两种情况，j个盘子装满和没装满，那么由于所有的苹果和盘子都是相同的，所以在盘子装满的情况下，其等于dp[i-1][j]，而在没装满的情况下，至少有一个盘子没满，那么就是dp[i][j-1]，因此：
# 状态转移方程为： dp[i][j] = dp[i-1][j] + dp[i][j-1]
m,n = 15, 6
dp = [[0 for i in range(n+1)]for j in range(m+1)]

for j in range(n+1):
    dp[0][j] = 1
for i in range(m+1):
    dp[i][0] = 0

for i in range(1, m+1):
    for j in range(1, n+1):
        dp[i][j] = dp[i][j-1] #如果j-1个盘子装满了，那再放这个苹果不会增加解法
        if i >= j: # 没装满的情况
            dp[i][j] += dp[i-j][j]

print(dp[m][n])


########## 递归
def fenApple(m, n):
    if m < 0 or n < 0:
        return 0
    elif m == 1 or n == 1:
        return 1
    else:
        return fenApple(m, n - 1) + fenApple(m - n, n)
print(fenApple(m, n))

