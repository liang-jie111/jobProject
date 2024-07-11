"""
author： liangjie
time：2024-07-09
"""
nums = [3,4,1,5,6,2,7]
stack = []
ans = [[-1, -1] for _ in range(len(nums))]

for i in range(len(nums)):
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