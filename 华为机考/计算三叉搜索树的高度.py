'''
定义构造三叉搜索树规则如下: 每个节点都存有一个数，当插入一个新的数时，从根节点向下寻找，直到找到一个合适的空节点插入查找的规则是:
    1.如果数小于节点的数减去500，则将数插入节点的左子树
    2.如果数大于节点的数加上500，则将数插入节点的右子树
    3.否则，将数插入节点的中子树
给你一系列数，请按以上规则，按顺序将数插入树中，构建出一棵三叉搜索树，最后输出树的高度。

输入描述
    第一行为一个数N，表示有N个数，1<=N<=10000
    第二行为N个空格分隔的整数，每个数的范围为[1,10000]
输出描述
    输出树的高度(根节点的高度为1)
'''
class Node:
    def __init__(self, val) -> None:
        self.val = val
        self.left = None
        self.mid = None
        self.right = None

    # 新数（节点）插入树中
    def insert(self, nval: int):
        if nval < self.val - 500:  # 插左子树
            if self.left:
                self.left.insert(nval)
            else:
                self.left = Node(nval)
        elif nval > self.val + 500:  # 插右子树
            if self.right:
                self.right.insert(nval)
            else:
                self.right = Node(nval)
        else:   # 插中子树
            if self.mid:
                self.mid.insert(nval)
            else:
                self.mid = Node(nval)

    # 获取树的高度
    def getHeight(self) -> int:
        maxHeight = 0
        if self.left:
            maxHeight = max(maxHeight, self.left.getHeight())
        if self.mid:
            maxHeight = max(maxHeight, self.mid.getHeight())
        if self.right:
            maxHeight = max(maxHeight, self.right.getHeight())
        return maxHeight + 1


n = 10
arr = [5000,2000,5000,8000,1400,1800,2000,500,1000,1300]
root = Node(arr[0])
for i in range(1, n):
    root.insert(arr[i])
print(root.getHeight())