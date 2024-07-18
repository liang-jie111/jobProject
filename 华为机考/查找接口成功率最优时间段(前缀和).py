'''
题目描述
    服务之间交换的接口成功率作为服务调用关键质量特性，某个时间段内的接口失败率使用一个数组表示。
    数组中每个元素都是单位时间内失败率数值，数组中的数值为0~100的整数，
    给定一个数值(minAverageLost)表示某个时间段内平均失败率容忍值，即平均失败率小于等于minAverageLost.找出数组中最长时间段，如果未找到则直接返回NULL。

输入描述
    有两行内容，第一行为 minAverageLost，第二行为数组，数组元素通过空格(" ")分隔,
    minAverageLost及数组中元素取值范围为0~100的整数，数组元素的个数不会超过100个

输出描述
    找出平均值小于等于minAverageLost的最长时间段，输出数组下标对，格式{beginIndex}-{endIndex} (下标从0开始)，
    如果同时存在多个最长时间段，则输出多个下标对且下标对之间使用空格(” “)拼接，多个下标对按下标从小到大排序。

示例1
    输入：
    1
    0 1 2 3 4
    输出：
    0-2

说明：
    A、输入解释：minAverageLost=1，数组[0, 1, 2, 3, 4]
    B、前3个元素的平均值为1，因此数组第一个至第三个数组下标，即0-2

解题思路：
    输入最小平均失败率和失败率数据。
    使用前缀和数组 psum 记录从头开始到每个位置的失败率累加和。
    使用两层循环遍历所有可能的时间段，通过前缀和数组判断平均失败率是否小于等于给定的阈值。
    如果找到一个满足条件的时间段，比较其长度是否大于当前已知的最大长度，更新最大长度和结果数组。
    最终输出所有满足条件的最长时间段。
'''
# 输入最小平均失败率
min_average_lost = 5

# 输入失败率数据
losts = [0, 1, 6, 10, 100, 8, 3, 6, 0, 19, 45, 67, 23, 1, 3]

n = len(losts)

# 计算前缀和
psum = [0] * (n + 1)
for i in range(n):
    psum[i + 1] = psum[i] + losts[i]

max_length = 1
result = []

# 寻找最长时间段
for l in range(n):
    for r in range(l + max_length - 1, n):
        length = r - l + 1
        # losts[l ~ r] 区间平均失败率小于等于 min_average_lost
        if psum[r + 1] - psum[l] <= min_average_lost * length:
            # 找到更长的时间段
            if length > max_length:
                max_length = length
                result.clear()
                result.append((l, r))
            elif length == max_length:  # 同时存在多个最长时间段
                result.append((l, r))

# 打印输出结果
for p in result:
    print(f"{p[0]}-{p[1]}", end=" ")





