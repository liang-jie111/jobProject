'''
题目描述
给定一个MxN的网格，其中每个单元格都填有数字，数字大小表示覆盖信号强度。灰色网格代表空地，橙色网格代表信号覆盖区域，绿色网格代表基站，绿色网格内数字大小表示该基站发射信号的初始强度。
基站信号每向外（上下左右）传播一格，信号强度减1，最小减为0，表示无信号，如下图示。当某个位置可以同时接收到多个基站的信号时，取其中接收信号强度的最大值作为该位置的信号强度。
对于给定网格，请判断是否存在一条路径，使得从左上角移动到右下角过程中信号不中断，只能上下左右移动。假设接收到的信号强度低于门限Th ，信号就会中断。
注意:出发点固定在网格的左上角，终点是网格的右下角。

输入描述
    第一行输入信号强度Th。 (1<= Th <= 100)
    第二行输入矩阵M、N。 (1<= M <= 100，1<= N <= 100)
    第三行输入基站个数K。 (1<= K <= 100)
    后续K行输入基站位置及初始信号强度。(前两个值表示基站所在行、列索引，第3个值表示基站初始信号强度)
输出描述
    返回信号不中断的最短路径，不存在返回0

示例1
输入：
1
4 4
2
0 1 2
3 2 3

输出：
6

解释:
1) 信号强度门限Th = 1
2) M=4，N=4
3) 4*4网格中一共包含2个基站
4) 2个基站的位置，其中第1个基站在第0行第1列、初始信号强度 =2;第2个基站在第3行第2列、初始信号强度=3
'''
import numpy as np

Th = 1

# m, n = 4, 4
# mat = np.array([[0]*n]*m)
# n_station = 2
# cor_station = [(0, 1), (3, 2)]
# degree_station = [2, 3]

m, n = 3, 3
mat = np.array([[0]*n]*m)
n_station = 2
cor_station = [(0, 1), (1, 1)]
degree_station = [2, 2]

def valid(r, c):
    return (0<= r < m) and (0<= c <n)

def spread(r, c):
    degree = mat[r][c]
    if degree > 1:
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            if (0<= r+dr < m) and (0<= c+dc <n) and (mat[r+dr][c+dc] < degree):
                mat[r+dr][c+dc] = max(degree-1, mat[r+dr][c+dc])
                spread(r+dr, c+dc)


for idx, (i,j) in enumerate(cor_station):
    mat[i][j] = degree_station[idx]
    spread(i, j)

print(mat)
visited = np.array([[0]*n]*m)
min_distance = m*n+1
flag = 0

def bfs(r, c, step):
    global min_distance, flag
    visited[r][c] = 1
    if r == m-1 and c == n-1:
        flag = 1
        min_distance = min(step, min_distance)
        visited[r][c] = 0
        return

    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        if (0 <= r + dr < m) and (0 <= c + dc < n) and (mat[r + dr][c + dc] >= Th) and visited[r + dr][c + dc] == 0:
            bfs(r+dr, c+dc, step+1)
    visited[r][c] = 0


if mat[0][0] < Th or flag == 0:
    print(0)
else:
    bfs(0, 0, 0)
    print(min_distance)