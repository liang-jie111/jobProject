# import sys

# for line in sys.stdin:
#     a = line.split()
#     print(int(a[0]) + int(a[1]))
need_allc = 1
used_memory = [(0,99)]
# print(used_memory)

start = 0
minDiff = 101
bestStart = -1
for m in used_memory:
    offset, length = m[0], m[1]
    if offset < start or length <= 0 or offset + length > 100:
        print(-1)
        exit(0)

    freeMem = offset - start
    if need_allc <= freeMem and freeMem - need_allc < minDiff:
        minDiff = freeMem - need_allc
        bestStart = start

    start = offset + length

if 100 - bestStart >= need_allc and 100 - start - need_allc < minDiff and start < 100:
    bestStart = start

print(bestStart)
