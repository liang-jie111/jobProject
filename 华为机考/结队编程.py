'''
题目描述
    某部门计划通过结队编程来进行项目开发，已知该部门有 N 名员工，每个员工有独一无二的职级，每三个员工形成一个小组进行结队编程，结队分组规则如下：
    从部门中选出序号分别为 i、j、k 的3名员工，他们的职级分别为 level[i]，level[j]，level[k]，结队小组满足 level[i] < level[j] < level[k] 或者 level[i] > level[j] > level[k]，其中 0 ≤ i < j < k < n。
    请你按上述条件计算可能组合的小组数量。同一员工可以参加多个小组。

输入描述
    第一行输入：员工总数 n
    第二行输入：按序号依次排列的员工的职级 level，中间用空格隔开

备注：
    1 <= n <= 6000
    1 <= level[i] <= 10^5
'''
n = 4
level = [1, 2, 3, 4]
# (1,2,3)、(1,2,4)、(1,3,4)、(2,3,4)
state = []
cnt = 0
res = []

def backtrack(start):
    global cnt
    if len(state) == 3:
        cnt += 1
        res.append(state.copy())
        return

    for i in range(start, n):
        backtrack(i + 1)
        if len(state) <= 1:
            state.append(level[i])
            backtrack(i+1)
            state.pop()
        elif (level[i] > state[1] and state[1] > state[0]) or (level[i] < state[1] and state[1] < state[0]):
            state.append(level[i])
            backtrack(i + 1)
            state.pop()
        else:
            pass

backtrack(0)
print(len(set([tuple(x) for x in res])))


