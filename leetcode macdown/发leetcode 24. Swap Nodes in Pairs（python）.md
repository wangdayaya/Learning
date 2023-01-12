leetcode  24. Swap Nodes in Pairs（python）

### 描述



Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)



Example 1:

![](https://assets.leetcode.com/uploads/2020/10/03/swap_ex1.jpg)
	Input: head = [1,2,3,4]
	Output: [2,1,4,3]
	
Example 2:
	
	Input: head = []
	Output: []


Example 3:


	Input: head = [1]
	Output: [1]


Note:

	The number of nodes in the list is in the range [0, 100].
	0 <= Node.val <= 100


### 解析

根据题意，就是给出了一个链表 head ，让我们在不对链表中各个节点的值进行修改的情况下，将两个相邻的节点进行互换，也就是第二个节点放到第一个节点之前，第四个节点放到第三个节点之前，先使用简单的方法，这种方法其实违背了题意，对链表的各个节点的值进行了修改，不推荐，思路如下：

* 如果只有一个节点或者没有节点，直接返回 head
* while 循环判断当 head 和 head.next 都不为空的情况下，（1）tail 保存第三个及之后的链表（2）使用 tmp 临时保存 head.val （3）将 head.next.val 赋予 head.val 表示将后一个节点的值赋给前一个节点（4）将 tmp 赋予 head.next.val 表示将前一个节点的值赋予后一个节点，这样两个节点的值就完成了互换
* 判断第三个和第四个节点是否都存在，如果是，则将 tail 赋予 head ，表示下一次循环交换第三个和第四个节点的值，否则直接返回 result



### 解答
				

	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def swapPairs(self, head):
	        """
	        :type head: ListNode
	        :rtype: ListNode
	        """
	        if not head or not head.next: return head
	        result = head
	        while head and head.next:
	            tail = head.next.next
	            tmp = head.val
	            head.val = head.next.val
	            head.next.val = tmp
	            if tail and tail.next:
	                head = tail
	            else:
	                break
	        return result
			
### 运行结果

	Runtime: 12 ms, faster than 96.91% of Python online submissions for Swap Nodes in Pairs.
	Memory Usage: 13.5 MB, less than 58.09% of Python online submissions for Swap Nodes in Pairs.

### 解析

我们可以用相同的解题方法，但是按照题意不修改节点值来完成题目。关键还是链表交换的基本操作要熟悉，：

	p->first->second->third 变成 p->second->first->third

运行结果可以看出，所占内存都减小显著。
### 解答

	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def swapPairs(self, head):
	        """
	        :type head: ListNode
	        :rtype: ListNode
	        """
	        if not head or not head.next: return head
	        result = p = ListNode(0)
	        p.next = head
	        while p.next and p.next.next:
	            third = p.next.next.next
	            second = p.next.next
	            first = p.next
	            p.next = second
	            p.next.next = first
	            p.next.next.next = third
	            if third:
	                p = p.next.next
	            else:
	                return result.next
	        return result.next

### 运行结果

	Runtime: 20 ms, faster than 60.06% of Python online submissions for Swap Nodes in Pairs.
	Memory Usage: 13.3 MB, less than 85.00% of Python online submissions for Swap Nodes in Pairs.

原题链接：https://leetcode.com/problems/swap-nodes-in-pairs/



您的支持是我最大的动力
