'''
给定一个二维数组M行N列，二维数组里的数字代表图片的像素，为了简化问题，仅包含像素1和5两种像素，每种像素代表一个物体2个物体相邻的格子为边界，求像素1代表的物体的边界个数。
像素1代表的物体的边界指与像素5相邻的像素1的格子，边界相邻的属于同一个边界，相邻需要考虑8个方向(上，下，左，右，左上，左下，右上，右下)。
其他约束:地图规格约束为:0<M<100,0<N<100
输入描述
    第一行，行数M， 列数N
    第二行开始，是M行N列的像素的二维数组，仅包含像素1和5
输出描述
    像素1代表的物体的边界个数。如果没有边界输出0(比如只存在像素1，或者只存在像素5)。

输入：
6  6
1	1	1	1	1	1
1	5	1	1	1	1
1	1	1	1	1	1
1	1	1	1	1	1
1	1	1	1	1	1
1	1	1	1	1	5

输出：
2
'''
def valid(r, c):
    # 验证坐标的有效性
    return 0 <= r < rows and 0 <= c < cols


def dfs(grid, r, c):
    # 深度优先搜索，标记为 0 的位置表示边界，然后搜索其周围的 0 的位置，并标记为 1。将所有相关联的位置都标记为 1。
    if not valid(r, c) or grid[r][c] != 0:
        return
    grid[r][c] = 1

    for dr in range(-1, 2):
        for dc in range(-1, 2):
            nr, nc = r + dr, c + dc
            dfs(grid, nr, nc)


if __name__ == "__main__":
    # 读取输入
    rows, cols = map(int, input().split())
    grid = [list(map(int, input().split())) for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                # 预先标记 grid[r][c] = 0，表示 (r, c) 为边界坐标
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if valid(nr, nc) and grid[nr][nc] == 1:
                            grid[nr][nc] = 0  # 标记为可能是边界

    result = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                # 使用深度优先搜索统计与边界相连的区域数量
                dfs(grid, r, c)
                result += 1

    # 输出结果
    print(result)