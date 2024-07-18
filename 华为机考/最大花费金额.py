'''
题目描述
    双十一众多商品进行打折销售，小明想购买自己心仪的一些物品，但由于受购买资金限制，所以他决定从众多心仪商品中购买三件，
    而且想尽可能的花完资金现在请你设计一个程序帮助小明计算尽可能花费的最大资金数额。

输入描述
    第一行为一维整型数组M，数组长度Q小于100，数组元素记录单个商品的价格,单个商品价格小于1000。
    第二行为购买资金的额度R，R小于100000。

输出描述
    输出为满足上述条件的最大花费额度
    如果不存在满足上述条件的商品，请返回-1。

示例1
    输入：
    23,26,36,27
    78
    输出：
    76

说明：
    金额23、26和27相加得到76，而且最接近且小于输入金额78。
'''
prices = [23, 26, 36, 27]
R = 78

prices.sort()
n = len(prices)

max_cost = -1
for i in range(n-2):
    for j in range(i + 1, n - 1):
        cost = prices[i] + prices[j]
        if cost + prices[j + 1] > R:  # 超过资金预算
            break

        # 二分尝试找到资金范围内最大的第三件商品
        left, right = j + 1, n
        while left + 1 < right:
            mid = (right + left) // 2
            if cost + prices[mid] <= R:
                left = mid
            else:
                right = mid

        # 购买 [i, j, left] 三件商品
        max_cost = max(max_cost, cost + prices[left])
print(max_cost)