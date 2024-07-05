a = [4, 1, 9, 8, 123, 0]


def bubble_sort(a):
    for i in range(0, len(a) - 1):
        for j in range(0, len(a) - 1 - i):
            if a[j] > a[j + 1]:
                tmp = a[j]
                a[j] = a[j + 1]
                a[j + 1] = tmp
    return a


print(bubble_sort(a))


def selection_sort(a):
    curr_min_index = 0
    for i in range(0, len(a)):
        curr_min_index = i
        for j in range(i + 1, len(a)):
            if a[j] < a[i]:
                curr_min_index = j
        tmp = a[curr_min_index]
        a[curr_min_index] = a[i]
        a[i] = tmp
    return a


print(selection_sort(a))


def insertion_sort(a):
    for i in range(0, len(a) - 1):
        for j in range(i, -1, -1):
            if a[j + 1] > a[j]:
                break
            else:
                tmp = a[j]
                a[j] = a[j + 1]
                a[j + 1] = tmp
    return a


#####################################################################################
'''
    堆排序
'''


def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1  # left = 2*i + 1
    r = 2 * i + 2  # right = 2*i + 2

    # 找到最大的节点
    if l < n and arr[i] < arr[l]:
        largest = l

    if r < n and arr[largest] < arr[r]:
        largest = r

    # 如果最大节点不是父节点，交换它们
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]

        # 递归调整最大节点
        heapify(arr, n, largest)


def heapSort(arr):
    n = len(arr)

    # 建立初始堆，从n/2节点处往前调整
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # 交换节点
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # 将堆顶元素与最后一个元素交换
        heapify(arr, i, 0)  # 将剩余的元素重新建堆


# 测试数据
arr = [10, 8, 12, 16, 14, 3, 7, 11, 18, 13]
heapSort(arr)
print("Sorted array is:", arr)

#####################################################################################
'''
    归并排序
'''


def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    left = merge_sort(left)
    right = merge_sort(right)
    return merge(left, right)


def merge(left, right):
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    return result


# 使用示例
arr = [5, 3, 4, 1, 2]
sorted_arr = merge_sort(arr)
print(sorted_arr)  # 输出: [1, 2, 3, 4, 5]

#####################################################################################
'''
    快速排序
'''


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


# 示例使用
arr = [3, 6, 8, 10, 1, 2, 1, 4, 7, 9]
sorted_arr = quicksort(arr)
print(sorted_arr)  # 输出: [1, 1, 2, 3, 4, 6, 7, 8, 9, 10]

#####################################################################################
'''
    希尔排序
'''


def shell_sort(arr):
    # 初始化增量序列
    n = len(arr)
    gap = n // 2
    while gap > 0:
        print("gap = ", gap)
        # 对每个增量进行插入排序
        for i in range(gap, n):
            print("i=", i)
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                print("  j=", j)
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
            print("   ", arr)
        print(arr)
        # 减小增量
        gap //= 2
        print("----------------")


# 示例使用
sample_list = [12, 34, 54, 2, 3, 87, 0, 19]
shell_sort(sample_list)
print("Sorted array is:", sample_list)

#####################################################################################
'''
    计数排序
'''


def counting_sort(arr):
    # 找到数组的最大值
    max_val = max(arr)
    # 计数数组，用于保存每个元素出现的次数
    count = [0] * (max_val + 1)
    # 统计每个元素的出现次数
    for num in arr:
        count[num] += 1

    # 将统计结果映射为排序后的数组
    sorted_arr = []
    for i in range(len(count)):
        sorted_arr.extend([i] * count[i])

    return sorted_arr


# 示例使用
arr = [4, 2, 9, 1, 7, 5, 8, 3, 6]
sorted_arr = counting_sort(arr)
print(sorted_arr)

#####################################################################################
'''
    基数排序
'''


def radix_sort(arr):
    # 获取最大数的位数
    max_num = max(arr)
    max_digit = len(str(max_num))

    for digit in range(max_digit):
        print("digit = ", digit)
        # 初始化计数排序的数组
        count_sort = [[] for _ in range(10)]

        # 提取当前位数
        for num in arr:
            place = (num // (10 ** digit)) % 10
            count_sort[place].append(num)

        # 将计数排序的结果重新组合
        arr = [num for count in count_sort for num in count]
        print(arr)
        print('-------')

    return arr


# 测试数据
arr = [170, 45, 75, 90, 11, 10, 20]
arr = radix_sort(arr)
print("Sorted array:", arr)
