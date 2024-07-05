"""
二叉树的层序遍历
"""
import collections
class Solution:
    def levelOrder(self, root):
        if not root: return []
        res, queue = [], collections.deque()
        queue.append(root)
        while queue:
            tmp = []
            for _ in range(len(queue)):
                node = queue.popleft()
                tmp.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            res.append(tmp)
        return res
'''
二叉树的锯齿形层序遍历
'''
class Solution:
    def zigzagLevelOrder(self, root):
        if not root: return []
        res, deque = [], collections.deque([root])
        while deque:
            tmp = collections.deque()
            for _ in range(len(deque)):
                node = deque.popleft()
                if len(res) % 2 == 0:
                    tmp.append(node.val) # 奇数层 -> 插入队列尾部
                else:
                    tmp.appendleft(node.val) # 偶数层 -> 插入队列头部

                if node.left:
                    deque.append(node.left)
                if node.right:
                    deque.append(node.right)
            res.append(list(tmp))
        return res


'''
二叉树的最大深度

一、DFS
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

二、BFS
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root: return 0
        queue, res = [root], 0
        while queue:
            tmp = []
            for node in queue:
                if node.left: tmp.append(node.left)
                if node.right: tmp.append(node.right)
            queue = tmp
            res += 1
        return res
'''

'''
二叉搜索树BST：一个二叉树中，任意节点的值要大于等于左子树所有节点的值，并且要小于等于右子树的所有节点的值

一、判断BST的合法性
def isValisBST(root, min, max):
    if root is None:
        return True
    if min and root.val < min.val:
        return False
    if max and root.val > max.val:
        return False    
    return isValidBST(root.left, min, root) and isValidBST(root.right, root, max)
    
isValisBST(root, null, null)


二、在BST中查找一个数是否存在
def isInBST(root, target):
    if root is None:
        return False
    if root.val == target:
        return True
    # return isInBST(root.left, target) or isInBST(root.right, target)
    if root.val > target:
        return isInBST(root.left, target)
    else:
        return isInBST(root.right, target)


三、在BST中插入一个数
def insertIntoBST(root, target):
    if root is None:
        return Node(target)
    if root.val == target:
        return root
    if root.val > target:
        root.left = insertIntoBST(root.left, target)
    if root.val < target:
        root.right = insertIntoBST(root.right, target)
        

四、在BST中删除一个数
def deleteNode(root, key):
    if root is None:
        return None
    if root.val == key:
        if root.left is None:  # 如果左子树为空，直接用右子树代替
            return root.right
        if root.right is None:  # 如果右子树为空，直接用左子树代替
            return root.left
         # 如果左右子树都不为空，则需要用左子树中最大的或者右子树中最小的
         minNode = getMin(root.right)
         root.val = minNode.val
         root.right = deleteNode(root.right, minNode.val)
        
    elif root.val > key:
        root.left = deleteNode(root.left, key)
    elif root.val < key:
        root.right = deleteNode(root.right, key)
'''

'''
计算完全二叉树的节点数，算法复杂度O(logNlogN)
    完全二叉树：每一层的节点都是紧凑靠左排列的
    满二叉树的高度和节点数呈指数关系，2^h - 1，其中h是树的高度

def countNodes(root):
    TreeNode l, TreeNode r = root, root
    # 记录左右子树的高度
    hl, hr = 0, 0
    while l != null:
        l = l.left
        hl += 1
    while r != null:
        r = r.right
        hr += 1
        
    if hl == hr:
        return math.pow(2, hl) - 1
        
    return 1 + countNodes(root.left) + countNodes(root.right)
'''


'''
Git原理之二叉树最近公共祖先

def lowestCommonAncestor(root, p, q):
    # base case
    if root is None:
        return None
    if root == p or root == q:
        return root
    TreeNode left = lowestCommonAncestor(root.left, p, q)
    TreeNode right = lowestCommonAncestor(root.right, p, q)
    
    # 情况1
    if left != null and right != null:
        return root
        
    # 情况2
    if left == null and right == null:
        return root
        
    # 情况3
    return left == null ? right : left
'''