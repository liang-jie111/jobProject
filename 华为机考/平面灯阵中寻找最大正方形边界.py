'''
题目描述
现在有一个二维数组来模拟一个平面灯阵，平面灯阵中每个位置灯处于点亮或熄灭，分别对应数组每个元素取值只能为1或0，现在需要找一个正方形边界，其每条边上的灯都是点亮(对应数组中元素的值为1)的，且该正方形面积最大。

输入描述
    第一行为灯阵的高度(二维数组的行数)
    第二行为灯阵的宽度(二维数组的列数)
    紧接着为模拟平台灯阵的二维数组arr
    1< arr.length <= 200 1< arr[0].length <= 200
输出描述
    返回满足条件的面积最大正方形边界信息。返回信息[r,c,w],其中r,c分别代表方阵右下角的行号和列号，w代表正方形的宽度。如果存在多个满足条件的正方形，则返回r最小的，若r相同，返回c最小的正方形。

输入：
4
5
1 0 0 0 1
1 1 1 1 1
1 0 1 1 0
1 1 1 1 1

输出：
[3,2,3]

解释：
满足条件且面积最大的正方形边界，其右下角的顶点为[3,2]，即行号为3，列号为2，其宽度为3，因此返回信息为[3,2,3]。

思路
从左上角开始，按照从左到右，从上到下的顺序，依[min(M,N), 1]从大到小，滑动判断是否有正方形，第一次遇到的正方形就是最大的且坐标满足条件的
'''
import numpy as np

# rows, cols = 4, 5
# mat = np.array([
#     [1, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1],
#     [1, 0, 1, 1, 0],
#     [1, 1, 1, 1, 1]
# ])


rows, cols = 3, 3
mat = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])




def isRec(r, c, w):
    # 判断以（r,c）为左上角坐标、边长为w的正方形满不满足条件
    # 上边
    for j in range(c, c+w):
        if mat[r][j] == 0:
            return False
    # 下边
    for j in range(c, c + w):
        if mat[r+w-1][j] == 0:
            return False
    # 左边
    for j in range(r, r + w):
        if mat[j][c] == 0:
            return False
    # 右边
    for j in range(r, r + w):
        if mat[j][c+w-1] == 0:
            return False

    # 如果四边都满足，返回True
    return True

max_w = min(rows, cols)

for w in range(max_w, 0, -1):
    # 从大到小找满足条件的矩形
    for r in range(rows - w + 1):
        for c in range(cols - w + 1):
            if isRec(r, c, w):
                print([r+w-1, c+w-1, w])
                exit(0)