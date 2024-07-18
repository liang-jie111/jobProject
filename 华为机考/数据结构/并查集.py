'''
题目描述
    总共有 n 个人在机房，每个人有一个标号 (1<=标号<=n) ，他们分成了多个团队，需要你根据收到的 m 条消息判定指定的两个人是否在一个团队中，具体的:
    消息构成为 a b c，整数 a、b 分别代表两个人的标号，整数 c 代表指令。
    c== 0 代表a和b在一个团队内。
    c == 1 代表需要判定 a 和b 的关系，如果 a和b是一个团队，输出一行"we are a team",如果不是，输出一行"we are not a team"。
    c 为其他值，或当前行a或b 超出 1~n 的范围，输出 "da pian zi"。

输入描述
    第一行包含两个整数 n，m(1<=n.m<=100000).分别表示有n个人和 m 条消息。
    随后的 m 行，每行一条消息，消息格式为: a b c (1<=a,b<=n, 0<=c<=1)
输出描述
    c ==1.根据 a 和 b 是否在一个团队中输出一行字符串,在一个团队中输出 "we are a team", 不在一个团队中输出 "we are not a team"。
    c 为其他值，或当前行 a 或 b 的标号小于 1 或者大于 n 时，输出字符串 "da pian zi"。
    如果第一行 n 和 m的值超出约定的范围时，输出字符串"NULL"。

示例1
输入
5 6
1 2 0
1 2 1
1 5 0
2 3 1
2 5 1
1 3 2

输出
we are a team
we are not a team
we are a team
da pian zi

https://zhuanlan.zhihu.com/p/93647900
'''
n, m = map(int, input().split())


def check_range(a: int, b: int, c=0) -> bool:
    return 1 <= a <= 100000 and 1 <= b <= 100000 and 0 <= c <= 1


if check_range(n, m):
    # 父节点数组
    father = [i for i in range(n + 1)]

    def find(x: int) -> int:
        if father[x] != x:
            # 一直找到最上层的父节点
            father[x] = find(father[x])
        return father[x]

    def merge(x: int, y: int):
        x_root, y_root = find(x), find(y)
        father[x_root] = y_root

    for _ in range(m):
        a, b, c = map(int, input().split())
        if check_range(a, b, c):
            if c == 0:
                merge(a, b)
            elif find(a) == find(b):
                print("we are a team")
            else:
                print("we are not a team")
        else:
            print("da pian zi")
else:
    print("NULL")




'''
题目描述
    为了达到新冠疫情精准防控的需要，为了避免全员核酸检测带来的浪费，需要精准圈定可能被感染的人群。
    现在根据传染病流调以及大数据分析，得到了每个人之间在时间、空间上是否存在轨迹的交叉。
    现在给定一组确诊人员编号(X1,X2,X3...Xn) 在所有人当中，找出哪些人需要进行核酸检测，输出需要进行核酸检测的人数。（注意:确诊病例自身不需要再做核酸检测)
    需要进行核酸检测的人，是病毒传播链条上的所有人员，即有可能通过确诊病例所能传播到的所有人。
    例如:A是确诊病例，A和B有接触、B和C有接触 C和D有接触，D和E有接触。那么B、C、D、E都是需要进行核酸检测的

输入描述
    第一行为总人数N
    第二行为确诊病例人员编号 (确证病例人员数量 < N) ，用逗号隔开
    接下来N行，每一行有N个数字，用逗号隔开，其中第i行的第个j数字表名编号i是否与编号j接触过。0表示没有接触，1表示有接触

输出描述
    输出需要做核酸检测的人数

补充说明
    人员编号从0开始
    0 < N < 100
    
示例1
输入：
5
1,2
1,1,0,1,0
1,1,0,0,0
0,0,1,0,1
1,0,0,1,0
0,0,1,0,1

输出
3

说明：
编号为1、2号的人员为确诊病例1号与0号有接触，0号与3号有接触，2号54号有接触。所以，需要做核酸检测的人是0号、3号、4号,总计3人要进行核酸检测。
'''
class UnionFind:
    def __init__(self, length):
        self.father = list(range(length + 1))

    def find(self, x):
        if not (0 <= x < len(self.father)):
            raise ValueError("查询越界")

        # 合并（路径压缩）
        if x != self.father[x]:
            self.father[x] = self.find(self.father[x])
        return self.father[x]

    def merge(self, x, y):
        x_root, y_root = self.find(x), self.find(y)
        self.father[y_root] = x_root


def main():
    n = int(input())
    confirm = list(map(int, input().split()))

    # 使用并查集将所有确诊的人员合并成一组
    start = confirm[0]
    uf = UnionFind(n)
    for i in range(1, len(confirm)):
        uf.merge(start, confirm[i])

    # 将所有有接触的人进行合并操作
    for i in range(n):
        row = list(map(int, input().split()))
        for j in range(len(row)):
            if row[j] == 1:
                uf.merge(i, j)

    cnt = 0  # 已经确认的总人数
    for i in range(n):
        if uf.find(i) == uf.find(start):
            cnt += 1

    # 输出， 这里排除了确诊病例自身不需要再做核酸检测人
    print(cnt - len(confirm))


if __name__ == "__main__":
    main()