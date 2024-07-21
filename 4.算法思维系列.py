import numpy as np
"""
1.求子集、求排列和求组合问题
    树，无非就是start或者contains剪枝
2.解数独
3.括号生成
    一类是判定括号合法性（栈），一类是合法括号的生成（回溯）
    
    合法性质：
        1.左括号数量一定等于右括号数量
        2.对于一个合法的括号字符串组合p，必然对于任何 0 <= i < len(p)都有：子串p[0：i]中左括号的数量都大于或等于右括号的数量

    问题改写：
        有2n个位置，每个位置可以放置'('或者')'，组成的所有括号组合中，有多少个是合法的？
4.BFS算法暴力破解各种智力问题
5.2Sum问题的核心思想
6.一个函数解决nSum问题
7.找出最长等值子数组
8.无重复字符的最长子串
9.前缀和解决子数组问题
10.下一个排列
11.在排序数组中查找元素的第一个和最后一个位置
"""
import numpy as np

################################################# 子集、组合、排列 #######################################
nums = [1, 2, 3]
'''
子集
输入一个不包含重复数字的数组，要求算法输出这些数字的所有子集
'''
def subsets(nums):
    '''
    思路一:[1,2,3]的子集可以由[1,2]的子集追加3得出
    '''
    if len(nums) == 0:
        return []
    n = nums.pop()
    res = subsets(nums)

    size = len(res)
    for i in range(size):
        res.append(res[i])
        res.append(n)
    return res


print(subsets(nums))


######################################
def backtrack(nums, start, state):
    res.append(state.copy())
    for i in range(start, len(nums)):
        state.append(nums[i])
        backtrack(nums, i+1, state)
        state.pop()
res = []
state = []
backtrack(nums, 0, state)
print(res)

'''
组合
输入两个数字n,k，算法输出[1..n]中k个数字的所有组合
'''
def combine(n, k):
    res = []
    state = []

    def backtrack(n, start, state):
        print(start, "##", state)
        if len(state) == k:
            res.append(state.copy())
            return
        # print(state, f"[{start}, {n}]")
        for i in range(start, n+1):
            state.append(i)
            backtrack(n, i+1, state)
            state.pop()
        print("    ")

    backtrack(n, 1, state)
    return res

print(combine(5, 2))

'''
排列
输入一个不包含重复数字的数组nums，返回这些数字的全部排列

组合和子集问题都有一个start变量防止重复，而排列问题中不需要防止重复
'''
def permute(nums):
    def backtrack(nums, state):
        if len(state) == len(nums):
            res.append(state.copy())
            return
        for i in range(0, len(nums)):
            if nums[i] in state:
                continue
            state.append(nums[i])
            backtrack(nums, state)
            state.pop()
    res = []
    state = []
    backtrack(nums, state)

    return res
print(permute([1,2,3,4,5]))


################################################# 解数独 #######################################
def solveSudoku(board):

    def isValid(board, row, col, num):
        '''判断board[i][j]能否放下数字num'''
        for i in range(0, 9):
            # 同行有没有重复数字
            if board[row][i] == num:
                return False
            # 同列有没有重复数字
            if board[i][col] == num:
                return False
            # 3*3 格子里有没有重复数字
            if board[(row//3)*3 + i//3][(col//3)*3 + i%3] == num:
                return False
        return True

    def backtrack(board, i, j):
        m, n = 9, 9
        if j == n:
            # 穷举到最后一列的话，换到下一行重新开始
            return backtrack(board, i+1, 0)
        if i == m:
            # 找到一个可行解
            print(board)
            return True

        if board[i][i] != ".":
            return backtrack(board, i, j+1)

        for ch in range(1, 10):
            if not isValid(board, i, j, str(ch)):
                continue
            board[i][j] = str(ch)

            if backtrack(board, i, j+1):
                return True

            board[i][j] = '.'
        return False
    return backtrack(board, 0, 0)


board = np.array([
    [".", ".", "2", ".", ".", "1", "9", "4", "."],
    [".", ".", ".", "8", "4", ".", ".", ".", "."],
    [".", "3", ".", ".", ".", "9", "5", "1", "."],
    [".", "4", ".", "9", ".", ".", "7", ".", "."],
    [".", ".", "8", ".", "7", "2", "3", ".", "."],
    [".", "6", ".", ".", ".", ".", ".", ".", "."],
    ["4", "2", ".", "7", "6", "8", "1", ".", "5"],
    [".", ".", "9", ".", "5", ".", ".", "6", "."],
    ["5", ".", "6", "1", ".", "3", "2", ".", "."]
])
print(solveSudoku(board))
print(board)
################################################# 括号生成 #######################################
'''
输入正整数n，输出是n对括号的所有合法组合

合法性质：
    1.左括号数量一定等于右括号数量
    2.对于一个合法的括号字符串组合p，必然对于任何 0 <= i < len(p)都有：子串p[0：i]中左括号的数量都大于或等于右括号的数量

问题改写：
    有2n个位置，每个位置可以放置'('或者')'，组成的所有括号组合中，有多少个是合法的？
'''
def genKuohao(n):
    def backtrack(left, right, state):
        '''
        :param left: 代表剩余左括号可用数量
        :param right: 代表剩余右括号可用数量
        '''
        # 数量小于0肯定不合法
        if left < 0 or right < 0:
            return
        # 若左括号剩下得多，说明不合法，返回
        if left > right:
            return
        # 当所有括号恰巧用完，得到一个合法的括号组合
        if left == 0 and right == 0:
            res.append(''.join((state.copy())))
            return

        state.append('(')
        backtrack(left-1, right, state)
        state.pop()

        state.append(')')
        backtrack(left, right-1, state)
        state.pop()

    res = []
    state = []
    if n == 0:
        return res
    backtrack(n, n, state)
    return res

print(genKuohao(3))

################################################# BFS算法暴力破解各种智力问题 #######################################

################################################# 2Sum问题的核心思想 #######################################
'''
无序数组的2Sum解法
'''
def twoSum(nums, target):
    # 一个dict，key为元素的值，value为元素值在原数组中的索引值
    index = {}
    for i in range(len(nums)):
        index[nums[i]] = i

    for i in range(len(nums)):
        other = target - nums[i]
        if other in index and index[other] != i:
            return [i, index[other]]
    return []
print(twoSum([3, 1, 5, 2, 4], 3))


'''
有多对儿目标和等于target，返回不重复的组合

思路：排序+双指针+去重
'''
def multiTwoSum(nums, target):

    nums_sort = np.sort(nums)
    index_sort = np.argsort(nums)

    low, high = 0, len(nums_sort) - 1
    res = []
    while low < high:
        left, right = nums_sort[low], nums_sort[high]
        sum = nums_sort[low] + nums_sort[high]
        if sum < target:
            low += 1
        elif sum > target:
            high -= 1
        else:
            res.append([index_sort[low], index_sort[high]])
            while low < high and nums_sort[low] == left:
                low += 1
            while low < high and nums_sort[high] == right:
                high -= 1
    return res

print(multiTwoSum([1,3,1,2,2,3,1], 4))


################################################# 一个函数解决nSum问题 #######################################
'''
3Sum问题
'''

def threeSum(nums, target):
    res = []
    nums_new = np.sort(nums)
    # 穷举threeSum的第一个数
    # for ...
    #   tuples = multiTwoSum()
    #   for tuple in tuples   如果存在二元组，那加上nums[i]就是三元组
    #   跳过第一个数字重复的情况，否则会出现重复结果
    return res


'''
nSum问题
'''
# 注意，nSum在调用之前就需要对nums排序了
def nSum(nums, n, start, target):
    # print(f"nums={nums}, n={n}, start={start}, target={target}")
    res = []
    size = len(nums)
    if n < 2 or size < n:
        return res
    if n == 2:
        low, high = start, size - 1
        while low < high:
            left, right = nums[low], nums[high]
            sum = nums[low] + nums[high]
            if sum > target:
                while low < high and nums[high] == right:
                    high -= 1
            elif sum < target:
                while low < high and nums[low] == left:
                    low += 1
            else:
                res.append([left, right])
                while low < high and nums[low] == left:
                    low += 1
                while low < high and nums[high] == right:
                    high -= 1
    else:
        for i in range(start, size):
            sub = nSum(nums, n-1, i+1, target - nums[i])
            for arr in sub:
                arr.append(nums[i])
                res.append(arr)
                # print(res)

            while i < size - 1 and nums[i] == nums[i+1]:
                i += 1

    return res

import numpy as np
nums = [-1, 4, 1, 2, 9, 0, 1, -1, 0, 6]
print(nSum(nums=np.sort(nums), n=5, start=0, target=0))


def remove_duplicates(lst):
    return [list(t) for t in set(tuple(l) for l in lst)]


################################################# 找出最长等值子数组 #######################################
'''
给你一个下标从 0 开始的整数数组 nums 和一个整数 k 。
如果子数组中所有元素都相等，则认为子数组是一个 等值子数组 。注意，空数组是 等值子数组 。
从 nums 中删除最多 k 个元素后，返回可能的最长等值子数组的长度。
子数组 是数组中一个连续且可能为空的元素序列。

nums = [1,3,2,3,1,3], k = 3
'''
from collections import defaultdict
def longestEqualSubarray(nums, k):
    res = 0
    cnt = defaultdict(int)
    print(cnt)
    left = 0   # 区间的left和right一定是最长子数组中最多的元素
    for right, x in enumerate(nums):
        cnt[x] += 1

        # 当前区间中，无法以 left = nums[i] 为等值元素构成合法等值数组
        while right - left + 1 - cnt[nums[left]] > k:
            cnt[nums[left]] -= 1
            left += 1
        res = max(res, cnt[x])

    return res

print(longestEqualSubarray([1,3,2,3,1,3], 3))

################################################# 无重复字符的最长子串 #######################################
'''
给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串的长度。
s = "abcabcbb"
'''
def lengthOfLongestSubstring(s):
    n = len(s)
    occ = set()
    right = 1
    max_length = 0
    for i in range(n):
        while right < n and s[right] not in s[i:right]:
        # while s[right] not in occ and right < n:
            # occ.add(s[right])
            right += 1
        max_length = max(max_length, right - i)
        if right == n:
            return max_length
        index = s.find(s[right])
        i = index + 1
    return max_length

print(lengthOfLongestSubstring('abcabcbb'))


################################################# 前缀和解决子数组问题 #######################################
'''
和为k的子数组

输入一个整数数组nums和一个整数k，算出nums中一共有几个和为k的子数组
'''
def subarraySum(nums, k):
    n = len(nums)
    sum = [0]*(n+1)
    for i in range(0, n):
        sum[i+1] = sum[i] + nums[i]

    print(sum)

    ans = 0
    for i in range(1, n+1):
        for j in range(0, i):
            if sum[i] - sum[j] == k:
                ans += 1

    return ans


nums = [3, 5, 2, -1, 4, -1]
print(subarraySum(nums, 2))

nums = [3, 5, 2, -2, 8, -1]
print(subarraySum(nums, 5))

################################################# 下一个排列 #######################################
'''
1.可以观察用例发现，只要从后往前找到第一个非升序排列的元素，即将要改变的元素,下标记作modify
2.因为要找到下一个排列，也就是比当前排列大且最接近的一个排列
3.那么我们将 要改变的元素 变为 比当前元素大且最接近的一个元素即可，将找到的这个元素下标记作target
4.因为要改变的元素对应位置之前的元素已经固定，所以我们要从要改变的元素之后的位置开始找
5.找到之后交换这两个位置的元素，那么交换之后，modify之后的元素从前往后看必定为降序序列
6.因为要找到最接近当前排列且大于当前排列的排列方式，那么modify之后的元素应该改为升序序列
7.所以最后一步将modify之后的元素倒序
'''
def nextPermutation(nums):
    """
    Do not return anything, modify nums in-place instead.
    """
    n = len(nums)
    if n <= 1:
        return

    # 倒序遍历，找到第一个相邻的升序序列, nums[i] > nums[i + 1]
    i = n - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1

    if i >= 0:
        # 倒序遍历，找到第一个大于nums[i]的元素，并将其与nums[i]交换位置
        j = n - 1
        while j >= 0 and nums[i] >= nums[j]:
            j -= 1
        # swap nums[i] and nums[j]
        nums[i], nums[j] = nums[j], nums[i]

    # 将nums[i + 1:]按照升序排列
    left, right = i + 1, n - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1

    return

################################################# 在排序数组中查找元素的第一个和最后一个位置 #######################################
'''
给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。
如果数组中不存在目标值 target，返回 [-1, -1]。
'''
def searchRange(nums, target):

    def search(target):
        l, r = 0, len(nums)
        while l < r:
            mid = (l + r) // 2
            if nums[mid] >= target:
                r = mid
            else:
                l = mid + 1
        return l

    # 首个target如果存在，一定是首个大于target-1的元素
    start = search(target)
    if start == len(nums) or nums[start] != target:
        return [-1, -1]  # 首个target不存在，即数组中不包含target
    # 找到首个大于target的元素，最后一个target一定是其前一位
    end = search(target + 1) - 1
    return [start, end]


print(searchRange([5,7,7,8,8,10], 8))

################################################# 排列 #######################################