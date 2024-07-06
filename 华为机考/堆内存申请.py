'''
题目描述
    有一个总空间为100字节的堆，现要从中新申请一块内存，内存分配原则为:
    优先分配紧接着前一块已使用的内存，分配空间足够时分配最接近申请大小的空闲内存。
输入描述
    第1行是1个整数，表示期望申请的内存字节数。
    第2到第N行是用空格分割的两个整数，表示当前已分配的内存的情况，每一行表示一块已分配的连续内存空间，每行的第1个和第2个整数分别表示偏移地址和内存块大小，如: 0 1 3 2 表示0偏移地址开始的1个字节和3偏移地址开始的2个字节已被分配，其余内存空闲。
输出描述
    若申请成功，输出申请到内存的偏移 若申请失败，输出-1。
    备注 1.若输入信息不合法或无效，则申请失败
        2.若没有足够的空间供分配，则申请失败
        3.堆内存信息有区域重叠或有非法值等都是无效输入
'''
start = 0
used_memory = [(0, 1), (2, 3), (7, 10)]
minFreeSize = 101
minStart = -1

needAlloc = 90

if needAlloc > 100 or needAlloc < 0:
    print(-1)

used_memory = sorted(used_memory, key=lambda x: x[0])

for offset, length in used_memory:
    if start > offset or start+needAlloc > 100 or offset + length > 100:
        print(-1)
        exit(0)

    if 0 < offset - start - needAlloc < minFreeSize:
        minFreeSize = offset - start - needAlloc
        minStart = start

    start = offset+length

if 0 < 100 - start - needAlloc < minFreeSize:
    minStart = start

print(minStart)