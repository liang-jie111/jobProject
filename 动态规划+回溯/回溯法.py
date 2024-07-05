from typing import List
'''
回溯法搜所有可行解的模板一般是这样的：

res = []
path = []

def backtrack(未探索区域, res, path):
    if path 满足条件:
        res.add(path) # 深度拷贝
        # return  # 如果不用继续搜索需要 return
    for 选择 in 未探索区域当前可能的选择:
        if 当前选择符合要求:
            path.add(当前选择)
            backtrack(新的未探索区域, res, path)
            path.pop()
'''

'''
组合
给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
你可以按 任何顺序 返回答案。
'''
class Solution:
    def combine(self, n: int, k: int):
        def backtrack(start):
            if len(state) == k:
                res.append(state.copy())
                return

            for i in range(start, n + 1):
                # if len(state)>0 and state[-1] > i:
                #     continue
                state.append(i)
                backtrack(i + 1)
                state.pop()

        state = []  # 状态（子集）
        start = 1  # 遍历起始点
        res = []  # 结果列表（子集列表）
        backtrack(start)
        return res


'''
组合总和
给定一个可能有重复数字的整数数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
candidates 中的每个数字在每个组合中只能使用一次，解集不能包含重复的组合。
'''

class Solution:
    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        def backtrack(
            state: list[int], target: int, choices: list[int], start: int, res: list[list[int]]
        ):
            """回溯算法：子集和 I"""
            # 子集和等于 target 时，记录解
            if target == 0:
                res.append(list(state))
                return
            # 遍历所有选择
            # 剪枝二：从 start 开始遍历，避免生成重复子集
            for i in range(start, len(choices)):
                # 剪枝一：若子集和超过 target ，则直接结束循环
                # 这是因为数组已排序，后边元素更大，子集和一定超过 target
                if target - choices[i] < 0:
                    break
                # 尝试：做出选择，更新 target, start
                state.append(choices[i])
                # 进行下一轮选择
                backtrack(state, target - choices[i], choices, i, res)
                # 回退：撤销选择，恢复到之前的状态
                state.pop()

        state = []  # 状态（子集）
        candidates.sort()  # 对 candidates 进行排序
        start = 0  # 遍历起始点
        res = []  # 结果列表（子集列表）
        backtrack(state, target, candidates, start, res)
        return res


'''
子集
给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的
子集（幂集）。
解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
'''
class Solution:
    def subsets(self, nums: list[int]) -> list[list[int]]:
        path = []
        ans = []
        n = len(nums)
        def dfs(i):
            if i == n:
                ans.append(path.copy())
                return
            dfs(i+1)

            path.append(nums[i])
            dfs(i+1)
            path.pop()
        dfs(0)
        return ans


'''
子集2
给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的 子集（幂集）。
解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。
'''
class Solution(object):
    def subsetsWithDup(self, nums):
        res, path = [], []
        nums.sort()
        self.dfs(nums, 0, res, path)
        return res

    def dfs(self, nums, index, res, path):
        res.append(path.copy())
        for i in range(index, len(nums)):
            if i > index and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            self.dfs(nums, i + 1, res, path)
            path.pop()


'''
单词搜索
给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
'''
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def dfs(i, j, k):
            if k == len(word) - 1:
                return board[i][j] == word[k]
            if board[i][j] != word[k]:
                return False
            c = board[i][j]
            board[i][j] = "0" # 此操作是防止在回溯中同一字母被反复使用
            # for a, b in itertools.pairwise((-1, 0, 1, 0, -1)):
            for a, b in [(-1,0),(0,1),(1,0),(0,-1)]:
                x, y = i + a, j + b
                if 0 <= x < m and 0 <= y < n and board[x][y] != "0" and dfs(x, y, k + 1):
                    return True
            board[i][j] = c
            return False

        m, n = len(board), len(board[0])
        return any(dfs(i, j, 0) for i in range(m) for j in range(n))


'''
复原 IP 地址
有效 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。
例如："0.1.2.201" 和 "192.168.1.1" 是 有效 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效 IP 地址。
给定一个只包含数字的字符串 s ，用以表示一个 IP 地址，返回所有可能的有效 IP 地址，这些地址可以通过在 s 中插入 '.' 来形成。你 不能 重新排序或删除 s 中的任何数字。你可以按 任何 顺序返回答案。
'''
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []
        path = []

        def backip(s, start):
            if len(path) > 4:  # 搜索路径大于4，进行剪枝
                return
            if len(path) == 4 and ''.join(path) == s:  # 搜索路径为4，且s被完全分割
                # if len(path) == 4 and start == len(s): #或者开始索引的位置是最后的s长度
                res.append('.'.join(path[:]))
            for i in range(start, len(s)):  # 从start 开始
                c = s[start:i + 1]  # 子串
                print(c, start, i)
                if c and 0 <= int(c) <= 255 and str(int(c)) == c:  # 判断是否满足ip地址
                    path.append(c)  # 加入当前path
                    backip(s, i + 1)
                    path.pop()  # 回溯

        backip(s, 0)
        return res


import sys
sys.stdout.write("0")