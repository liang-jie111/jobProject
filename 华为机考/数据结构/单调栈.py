"""
author： liangjie
time：2024-07-09
"""
nums = [3,4,1,5,6,2,7]


##################################### 解法一，分两个方向遍历，分别计算两个方向的，最后合并
def nextSmallerNum(nums):
    stack = []
    n = len(nums)
    res = [-1]*n
    for i in range(n-1, -1, -1):
        while len(stack) > 0 and nums[stack[-1]] >= nums[i]:
            stack.pop()
        res[i] = -1 if len(stack) == 0 else stack[-1]
        stack.append(i)

    return res

ans1 = nextSmallerNum(nums)
ans2 = nextSmallerNum(nums[::-1])
ans = [[len(nums)-a2-1 if a2!=-1 else -1, a1] for a2,a1 in zip(ans2[::-1], ans1)]
print(ans)


##################################### 解法二，遍历时同时计算前后方向
nums = [3,4,1,5,6,2,7]
stack = []
ans = [[-1, -1] for _ in range(len(nums))]

for i in range(len(nums)):
    # 栈里必须是比当前元素更小的元素，且栈里元素是单调递增的，所以这个判断可更新栈中元素的右侧值
    while stack and nums[stack[-1]] > nums[i]:
        ans[stack[-1]][1] = i
        stack.pop()
    if stack:
        if nums[stack[-1]] < nums[i]:
            ans[i][0] = stack[-1]
        elif nums[stack[-1]] == nums[i]:
            ans[i][0] = ans[stack[-1]][0]
    stack.append(i)

print(ans)