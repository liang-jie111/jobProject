def func(n):
    l, r = 0, n
    error = 0.00001
    mid = (l+r)/2
    while 1:
        if mid*mid < n and n - mid*mid > error:
            l = mid
            mid = (l+r)/2
        if mid*mid > n:
            r = mid
            mid = (l+r)/2
        if n - mid*mid <= error and n >= mid*mid:
            return mid
        print(l, ' ', mid, ' ', r)
        print(n - mid*mid <= error)
        print(n >= mid*mid)

print(func(3))
