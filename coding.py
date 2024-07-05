import re

import numpy as np

# def backtrack():
#     if len(state)==n:
#         res.append(state.copy())
#         return
#     for word in ['c', 'o', 'w']:
#         # 判断当前Word是否能添加到state中
#         if len(state)<=1 or state[-2:]+[word] != ['c', 'o', 'w']:
#             state.append(word)
#             backtrack()
#             state.pop()
# n=3
# res = []
# state = []
# backtrack()
#
# print(len(res))
# print(res)



# d = {1:2, 2:5, 0:4}
# d.setdefault(0, 0)
# d[0] += 1
# sorted_dict = dict(sorted(d.items(), key=lambda x: x[0]))

# # 假设有以下字典
# d = {'a': 3, 'b': 1, 'd': 2, 'c':2}
# # 首先根据值排序
# sorted_by_value = sorted(d.items(), reverse=True, key=lambda item: (item[1], item[0]))
# print(sorted_by_value)
# print(format(0.263434, '.1f'))


# import bisect  # 导入查找模块
# def get_max_sub(arr):  # 定义获取最长子序列函数
#     res = [arr[0]]  # 将传入的列表第一个参数放入res
#     dp = [1] * len(arr)  # 定义一个长度为输入列表长度的列表，元素为1.
#     for i in range(1, len(arr)):  # 计算以arr[i]结尾的最长上升子序列长度
#         if arr[i] > res[-1]:  # 如果arr[i]大于最后一个元素，插入
#             res.append(arr[i])
#             dp[i] = len(res)
#         else:  # 如果arr[i]小于最后一个元素，找到res中比他大的元素的位置，并将该元素替换为arr[i]
#             index = bisect.bisect_left(res, arr[i])
#             res[index] = arr[i]
#             dp[i] = index + 1
#     return dp
#
# def get_max_sub2(arr):
#     n = len(arr)
#     dp = [0 for _ in range(n)]
#     for i in range(n):
#         dp[i][i] = 1
#
#     for i in range(1,n):
#         for j in range(i+1, n):
#             if arr[j] > arr[j-1]:
#                 dp[i][j] = dp[i][j-1] + 1
#             else:
#                 dp[i][j] = dp[i][j-1]
#     print(np.array(dp))
#     res = []
#     for i in range(n):
#         res.append(max(dp[:][i]))
#     return res
#
#
# dp = get_max_sub([1,1,3,2,4,5,3,8])
# dp2 = get_max_sub2([1,1,3,2,4,5,3,8])
# print(dp)
# print(dp2)

list1 = [1, 1, 2, 3, 5, 6]
list2 = [1, 3, 6]

# 创建一个集合来跟踪 list2 中的元素，以便快速查找
set2 = set(list2)

# 标记哪些元素需要被移除（使用 None 作为标记）
marked = [None if item in set2 else item for item in list1]

# 创建一个新的列表，其中移除了被标记为 None 的元素
result = [item for item in marked if item is not None]

print(result)  # 输出: [1, 2, 3, 5]