'''
快排非递归 python 快速排序的非递归实现
模拟栈操作实现非递归的快速排序
非递归的实现快速排序，可以利用栈来实现。其实递归的本质就是栈的先进后出的性质。
    利用快速排序每次是递归向左右子序列中进行排序，利用栈我们可以把左右子序列的端点值保存到栈中，然后每次取栈顶区间进行排序，直到栈为空则整个序列为有序
'''

def QuickSort(arr):
    if len(arr) < 2:
        return arr
    stack = []
    stack.append([0, len(arr)-1])   #初始化桟
    while stack:
        l, r = stack.pop()   #出栈一个区间
        index = partition(arr, l, r)    #对分区进行一趟交换操作，并返回基准线下标
        if l < index - 1:
            stack.append([l, index - 1]) #当前趟，左区间入栈
        if r > index + 1:
            stack.append([index + 1, r]) #当前趟，右区间入栈


def partition(arr, left, right):   # 对分区进行一趟交换操作，并返回基准线下标
    tmp = arr[left]    # 用区间的第1个记录作为基准
    while left < right:
        while left < right and arr[right] >= tmp:
            right -= 1
        arr[left] = arr[right]
        while left < right and arr[left] <= tmp:
            left += 1
        arr[right] = arr[left]
    #此时s゠t
    arr[left] = tmp
    return left


nums = [4, 5, 0, -2, -3, 1]
QuickSort(nums)
print(nums)