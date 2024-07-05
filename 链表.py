"""
判断链表中是否有环，且返回环的入口

假设入口处距离head为a，换中个数为b，即k=a+b。
则每次相遇，fast比slow多走nb步
F = 2S
F = S + nb
上两式得出：
F = 2nb S=nb
又知每次走到入口节点处必走a+mb步，此时slow节点已经走了nb步，再走a步即可到达入口点处，那么当第一次相遇后让fast节点返回head节点，
此时fast和slow都走1步，那么相遇时fast节点走了a步，slow节点走了a+nb步，根据分析，此时相遇的节点必为环的入口节点
"""
class Solution(object):
    def detectCycle(self, head):
        fast, slow = head, head
        while True:
            if not (fast and fast.next): return
            fast, slow = fast.next.next, slow.next
            if fast == slow: break
        fast = head
        while fast != slow:
            fast, slow = fast.next, slow.next
        return fast
