# 给定一个m * n的整数矩阵作为地图，短阵数值为地形高度； 中庸行者选择地图中的任意一点作为起点，尝试往上、下、左、右四个相邻格子移动; 移动时有如下约束:
#
# 中庸行者只能上坡或者下坡，不能走到高度相同的点
# 不允许连续上坡或者连续下坡，需要交替进行，
# 每个位置只能经过一次，不能重复行走，
# 请给出中庸行者在本地图内，能连续移动的最大次数。
import numpy as np

def valid(r, c):
    return r>=0 and r<m and c >=0 and c<n and visited[r][c]==0

def dfs(r, c, step, flag):
    global max_step
    visited[r][c] = 1
    max_step = max(max_step, step)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if valid(r + dr, c + dc) and mat[r][c] != mat[r+dr][c+dc]:
            if -flag == 1 and mat[r][c] > mat[r+dr][c+dc]:
                continue
            if -flag == -1 and mat[r][c] < mat[r+dr][c+dc]:
                continue
            dfs(r + dr, c + dc, step+1, -flag)
    visited[r][c] = 0

m, n = 4,4
mat = np.array([
    [1,8,2,5],
    [4,6,2,9],
    [1,8,3,6],
    [9,7,5,1]
])
max_step = 0
visited = np.array([[0]*n]*m)
for r in range(m):
    for c in range(n):
        # 每个点出发，都从大或小开始
        dfs(r, c, 0, 1)
        dfs(r, c, 0, -1)

print(max_step)