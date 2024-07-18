"""
0-1背包问题
问题描述：有一个背包可以装物品的总重量为W，现有N个物品，每个物品重w[i]，价值v[i]，用背包装物品，能装的最大价值是多少？

对于这种问题，我们一般采用动态规划的方法来进行解决。我们定义动规数组f[i][j]来表示前i件物品，容量为j时的最大价值，则
if    j≥w[i], 则f[i][j]= max(f[i−1][j],  f[i−1][j−w[i]]+v[i])
else  f[i][j]= f[i−1][j]
"""
c = 10
w = [2,2,6,5,4]
v = [6,3,5,4,6]
n = 5
dp = [[0]*(c+1) for _ in range(n)]

for i in range(1, c+1):
    if i >= w[0]:
        dp[0][i] = v[0]

for i in range(1, n):
    for j in range(1, c + 1):
        if j >= w[i]:
            dp[i][j] = max(dp[i-1][j], dp[i-1][j-w[i]] + v[i])
        else:
            dp[i][j] = dp[i-1][j]

print(dp[n-1][c])

"""
1、从m个物品中选出若干个，其总价低于N
2、物品分为主件与附件，如果要买为附件的物品，必须先买该附件所属的主件
3、使得每件物品的价格与重要度的乘积的总和最大

在本题中进行了一项变动，即物品分为主件和附件，考虑到一个主件最多可以购买两个附件，那我们可以细化分析，将是否购买该物品，细化为是否购买该物品，以及是否购买该物品的附件，即5种情况，
不购买该物品，购买该物品，购买该物品及附件1，购买该物品及附件2，购买该物品及附件1及附件2，f[i][j]取这五种情况的最大值，这五种情况分别对应于:
f[i-1][j],
f[i-1][j-w[i]]+v[i],
f[i][j] = max(f[i-1][j], f[i-1][j-w[i]-w[a1]]+v[i]+v[a1]),
f[i][j] = max(f[i-1][j], f[i-1][j-w[i]-w[a2]]+v[i]+v[a2]),
f[i][j] = max(f[i-1][j], f[i-1][j-w[i]-w[a1]-w[a2]]+v[i]+v[a1]+v[a2]),
其中a1和a2是该物品的附件。
"""
n, m = map(int, input().split())
primary, annex = {}, {}
for i in range(1, m+1):
    x, y, z = map(int, input().split())
    if z==0:#主件
        primary[i] = [x, y]
    else:#附件
        if z in annex:#第二个附件
            annex[z].append([x, y])
        else:#第一个附件
            annex[z] = [[x,y]]
m = len(primary)#主件个数转化为物品个数
dp = [[0]*(n+1) for _ in range(m+1)]
w, v= [[]], [[]]
for key in primary:
    w_temp, v_temp = [], []
    w_temp.append(primary[key][0])#1、主件
    v_temp.append(primary[key][0]*primary[key][1])
    if key in annex:#存在主件
        w_temp.append(w_temp[0]+annex[key][0][0])#2、主件+附件1
        v_temp.append(v_temp[0]+annex[key][0][0]*annex[key][0][1])
        if len(annex[key])>1:#存在两主件
            w_temp.append(w_temp[0]+annex[key][1][0])#3、主件+附件2
            v_temp.append(v_temp[0]+annex[key][1][0]*annex[key][1][1])
            w_temp.append(w_temp[0]+annex[key][0][0]+annex[key][1][0])#3、主件+附件1+附件2
            v_temp.append(v_temp[0]+annex[key][0][0]*annex[key][0][1]+annex[key][1][0]*annex[key][1][1])
    w.append(w_temp)
    v.append(v_temp)
for i in range(1,m+1):
    for j in range(10,n+1,10):#物品的价格是10的整数倍
        max_i = dp[i-1][j]
        for k in range(len(w[i])):
            if j-w[i][k]>=0:
                max_i = max(max_i, dp[i-1][j-w[i][k]]+v[i][k])
        dp[i][j] = max_i
print(dp[m][n])

''' 空间优化 '''
n, m = map(int,input().split())
primary, annex = {}, {}
for i in range(1,m+1):
    x, y, z = map(int, input().split())
    if z==0:
        primary[i] = [x, y]
    else:
        if z in annex:
            annex[z].append([x, y])
        else:
            annex[z] = [[x,y]]
dp = [0]*(n+1)
for key in primary:
    w, v= [], []
    w.append(primary[key][0])#1、主件
    v.append(primary[key][0]*primary[key][1])
    if key in annex:#存在附件
        w.append(w[0]+annex[key][0][0])#2、主件+附件1
        v.append(v[0]+annex[key][0][0]*annex[key][0][1])
        if len(annex[key])>1:#附件个数为2
            w.append(w[0]+annex[key][1][0])#3、主件+附件2
            v.append(v[0]+annex[key][1][0]*annex[key][1][1])
            w.append(w[0]+annex[key][0][0]+annex[key][1][0])#4、主件+附件1+附件2
            v.append(v[0]+annex[key][0][0]*annex[key][0][1]+annex[key][1][0]*annex[key][1][1])
    for j in range(n,-1,-10):#物品的价格是10的整数倍
        for k in range(len(w)):
            if j-w[k]>=0:
                dp[j] = max(dp[j], dp[j-w[k]]+v[k])
print(dp[n])