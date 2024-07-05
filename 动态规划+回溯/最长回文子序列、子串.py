"""
dp[i][j]含义：在子串s[i:j]中，最长回文子序列的长度为dp[i][j]
"""
import numpy as np


def longestPalindSubSeq(s):
    n = len(s)

    '''
    i肯定小于等于j，对于那些i>j的位置，根本不存在子序列，初始化为0
    '''
    # dp = [[0]*n]*n
    dp = np.array([[0]*n]*n)

    '''
    base case:如果只有一个字符，最长回文子序列长度为1
    '''
    for i in range(n):
        dp[i][i] = 1

    for i in range(n-2, -1, -1):
        for j in range(i+1, n):
            if s[i] == s[j]:
            # if s[i] == s[j] and dp[i+1][j-1] != 0:
                '''
                如果相等，则它俩一定在回文子串中
                '''
                dp[i][j] = dp[i+1][j-1]+2
            else:
                '''
                如果不相等，说明他俩不可能同时出现在s[i:j]的最长回文子序列中，那么把他俩分别加入s[i+1:j-1]中，看看哪个的更长即可
                '''
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    print(dp)
    return dp[0][n - 1]

s = 'LONNNNP'
print(longestPalindSubSeq(s))

'''
最长回文子串
'''
str = input()
n = len(str)
list = []
for i in range(0,n-1):
    for j in range(i+1,n):
        if str[j] == str[i] and str[i+1:j] == str[j-1:i:-1]:
            list.append(len(str[i:j+1]))
print(max(list))
