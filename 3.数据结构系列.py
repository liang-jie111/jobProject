######################################################  单调栈  ########################################################

'''
下一个更大的元素
给定一个数组，返回一个等长数组，对应索引存着下一个更大的元素，如果没有更大元素，存-1
'''
def nextGreaterElement(nums):
    stack = []
    n = len(nums)
    ans = [-1]*n
    for i in range(n-1, -1, -1):
        print(i)
        while len(stack) > 0 and stack[-1] <= nums[i]:
            stack.pop()
        print(f"i={i}, stack = {stack}")
        ans[i] = -1 if len(stack) == 0 else stack[-1]
        stack.append(nums[i])
    return ans

nums = [2,1,2,4,3,5]
print(nextGreaterElement(nums))

'''
数组T存放的是近几天的温度，返回一个数组，计算对于每一天，还要至少等多少天才能等到一个更暖和的气温；如果等不到那一天，填0
'''
def getNextWarmerDay(T):
    stack = []
    n = len(T)
    ans = [-1] * n
    for i in range(n - 1, -1, -1):
        while len(stack) > 0 and T[stack[-1]] <= T[i]:
            stack.pop()
        ans[i] = 0 if len(stack) == 0 else (stack[-1] - i)
        stack.append(i)
    return ans

T = [73,74,75,71,69,72,76,73]
print(getNextWarmerDay(T))

'''
下一个更大的元素2
给定一个循环数组，返回一个等长数组，对应索引存着下一个更大的元素，如果没有更大元素，存-1
'''
def nextGreaterElement2(nums):
    stack = []
    n = len(nums)
    ans = [-1]*n
    for i in range(2*n-1, -1, -1): # 假装数组长度翻倍，那每个元素就可以看到自己前面的元素了
        print(i)
        while len(stack) > 0 and stack[-1] <= nums[i % n]:
            stack.pop()
        ans[i % n] = -1 if len(stack) == 0 else stack[-1]
        stack.append(nums[i%n])
    return ans

nums2 = [2,1,2,4,3]
print(nextGreaterElement2(nums2))

######################################################  回文链表  ########################################################
'''寻找回文串是从中间向两端扩展，判断回文串是从两端向中间收缩'''

"""
判断回文单链表
思路一、链表递归实现
    链表兼具递归结构，树结构不过是链表的衍生，所以链表也有前序和后序遍历，模板如下：
    traverse(head):
        if head==null: return
        traverse(head.next)
        # 后续遍历操作代码
        ...
    算法的时间和空间复杂度都是O(N)
思路二、优化空间复杂度，快慢指针，找中点，从两端向中间逐个判断
    空间复杂度为O(1)
    1.先通过快慢指针找到链表的中点
        注意：节点数为奇数，fast指针没有指向null，fast停止移动后，需要把slow再往后挪一位
    2.从slow开始，翻转后面的链表，可以开始比较
"""
class ListNode:
    def __init__(self, val, next):
        self.val = val
        self.next = next
############### 思路一
global left

def isPalindrome(head):
    global left
    left = head
    return traverse(head)

# 利用递归，倒序遍历单链表
def traverse(right):
    global left
    if right is None:
        return True
    res = traverse(right.next)
    # 后续遍历操作
    res = res and (right.val == left.val)
    left = left.next
    return res

############### 思路二
def reverse(head):
    '''
    标准的翻转链表算法
    '''
    pre, cur = None, head
    while cur is not None:
        next = cur.next
        cur.next = pre
        pre = cur
        cur = next
    return pre

def isPalindrome2(head):
    slow, fast = head, head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
    if fast is not None:
        slow = slow.next

    left = head
    right = reverse(slow)
    while right is not None:
        if left.val != right.val:
            return False
        left = left.next
        right = right.next
    return True


######################################################  递归翻转链表  ########################################################
'''
递归翻转整个链表
'''
def reverse2(head):
    if head is None or head.next is None:
        return head
    last = reverse2(head.next)
    # 后续遍历操作：从后往前翻转节点
    head.next.next = head
    head.next = None
    return last

'''
翻转链表前N个节点
'''
global successor
def reverseTopK(head, k):
    global successor
    if k == 1:
        successor = head.next
        return head
    last = reverseTopK(head.next, k - 1)
    head.next.next = head
    # 让翻转之后的head节点和后面的节点连起来
    head.next = successor
    return last

'''
翻转链表的一部分
'''
def reversePart(head, m, n):
    if m == 1:
        return reverseTopK(head, n)
    # 对于head.next来说，就是翻转区间[m-1,n-1]，前进到翻转的起点出发base case
    head.next = reversePart(head.next, m-1, n-1)
    return head


######################################################  k个一组翻转链表  ########################################################
'''
k个一组翻转链表，剩余不足k个的节点保持原有顺序

递归性质：函数只需要关心如何翻转前k个节点，然后递归调用自己即可，因为子问题和原问题的结构完全相同，这就是所谓的递归性质

1.先翻转以head开头的k个元素
2.将第k+1个元素作为head递归调用函数
3.将上述两个过程的结果连接起来
4.base case是剩余不足k个元素的时候直接返回
'''
def reverseA2B(start, end):
    pre, cur, next = None, start, start
    while cur != end:
        next = cur.next
        cur.next = pre
        pre = cur
        cur = next
    return pre

def reverseGroupK(head, k):
    # 标准的翻转链表算法相当于是"翻转节点a到null之间的节点"，现在是"反转a到b之间的节点"
    if head is None:
        return None
    # 区间[a, b)包含k个待翻转的元素
    a, b = head, head
    for i in range(k):
        if b is None:
            return head
        b = b.next
    # 翻转前k个元素
    newHead = reverseA2B(a, b)

    # 递归翻转后续链表并连接起来
    a.next = reverseGroupK(b, k)
    return newHead