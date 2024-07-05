def merge(seq, low, mid, high):
    """合并两个已排序好的列表，产生一个新的已排序好的列表"""
    # 通过low,mid height 将[low:mid) [mid:height)提取出来
    left = seq[low:mid]
    right = seq[mid:high]
    print('left:', left, 'right:', right)

    k = 0  # left的下标
    j = 0  # right的下标
    result = []  # 保存本次排序好的内容
    # 将最小的元素依次添加到result数组中
    while k < len(left) and j < len(right):
        if left[k] <= right[j]:
            result.append(left[k])
            k += 1
        else:
            result.append(right[j])
            j += 1
    # 将对比完后剩余的数组内容 添加到已排序好数组中
    result += left[k:]
    result += right[j:]
    # 将原始数组中[low:high)区间 替换为已经排序好的数组
    seq[low:high] = result
    print("seq:", seq)


seq = [5, 4, 3, 0, 1, 2, 7, 5, 11, 9]
i = 1 # group length，从每组只有1个数字开始
while i < len(seq):
    print('子数组 长度 : ', i)
    low = 0
    while low < len(seq):
        mid = low + i
        high = min(low + 2 * i, len(seq))
        if mid < high:
            print('low ', low, 'mid:', mid, 'high:', high)
            merge(seq, low, mid, high)
            print('\n')
        low += 2*i
    i *= 2
