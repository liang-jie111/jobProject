'''
题目描述
    某云短信厂商，为庆祝国庆，推出充值优惠活动。现在给出客户预算，和优惠售价序列，求最多可获得的短信总条数。

输入描述
    第一行客户预算M，其中 0<=M<=100
    第二行给出售价表，P1,P2,... Pn, 其中 1<=n<=100
    Pi为充值i元获得的短信条数.
    1<=Pi<=1000,1<=n<=100
输出描述
    最多获得的短信条数。

示例1
    输入
        6
        10 20 30 40 60
    输出
        70

说明
    分两次充值最优，1元、5元各充一次。总条数10+60=70
'''
M = 18
p = [1, 2, 30, 40, 60, 84, 70, 80, 90, 150]

# dp[x] 表示充值 x 元最多获得的短信条数
dp = [0] * (M + 1)
length = len(p)
for i in range(1, length + 1):  # 物品
    cnt = p[i - 1]  # 充值 i 元获得的短信条数
    for j in range(i, M + 1):  # 容量
        dp[j] = max(dp[j], dp[j - i] + cnt)

print(dp[M])



nums = [5,4,2,3,2,4,9]
capacity = 10
# dp[x] 表示容量为x时有几种坐满的方式
dp = [0] * (capacity + 1)
dp[0] = 1

for x in nums:
    for cap in range(capacity, x - 1, -1):
        dp[cap] += dp[cap - x]

print(dp[capacity])
