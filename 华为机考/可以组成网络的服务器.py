'''
在一个机房中，服务器的位置标识在n*m的整数矩阵网格中，1表示单元格上有服务器，0表示没有。如果两台服务器位于同一行或者同一列中紧邻的位置，
则认为它们之间可以组成一个局域网，请你统计机房中最大的局域网包含的服务器个数。
输入描述
    第一行输入两个正整数，n和m，0<n,m<=100
    之后为n*m的二维数组，代表服务器信息
输出描述
    最大局域网包含的服务器个数。
'''
import numpy as np

n, m = 5, 5
mat=np.array([
    [1,0,0,1,0],
    [1,1,0,1,0],
    [0,1,1,1,0],
    [1,0,0,0,1],
    [1,1,1,1,1]
])
visited = np.array([[0]*m]*n)

def valid(r, c):
    return 0<=r<n and 0<=c<m and visited[r][c]==0

def dfs(r, c, step):
    global max_res
    # print(r, c)
    visited[r][c] = 1
    cnt = 1
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if not valid(r + dr, c + dc) or mat[r + dr][c + dc] != 1:
            continue
        cnt += dfs(r+dr, c+dc, step+1)
    return cnt


max_res = 0
for i in range(n):
    for j in range(m):
        if visited[i][j] == 0 and mat[i][j] == 1:
            max_res = max(max_res, dfs(i, j, 1))

print(max_res)