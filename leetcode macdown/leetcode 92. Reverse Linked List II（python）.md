leetcode  92. Reverse Linked List II（python）

### 描述

Given the head of a singly linked list and two integers left and right where left <= right, reverse the nodes of the list from position left to position right, and return the reversed list.

Follow up: Could you do it in one pass?



Example 1:

![](https://assets.leetcode.com/uploads/2021/02/19/rev2ex2.jpg)

	Input: head = [1,2,3,4,5], left = 2, right = 4
	Output: [1,4,3,2,5]


	
Example 2:

	Input: head = [5], left = 1, right = 1
	Output: [5]






Note:

	The number of nodes in the list is n.
	1 <= n <= 500
	-500 <= Node.val <= 500
	1 <= left <= right <= n


### 解析

根据题意，给定一个单向链表的头部 head 和左右两个整数 left 和 right ，其中 left <= right ，将列表的节点从左位置反转到右位置，并返回反转后的列表。题目还给有能力的同学提出了更高的要求，能否一次性遍历完成。

这道题就是考察链表的反转、拼接、遍历等基本操作，我们如果是在比赛中，看到题目对于节点数量和节点值的限制范围很小，所以直接将所有的节点值存入列表中，然后对 [left，right] 中的值进行反转，然后再遍历列表中的值，拼接成一个新的链表即可。

但是这种方法比较投机取巧，对于节点数很多的链表就会超时或者使用内存过高了。所以题目中才会提出能够一次遍历就转换这个链表的要求。


### 解答
				
	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def reverseBetween(self, head, left, right):
	        if not head or not head.next: return head
	        values = []
	        while head:
	            values.append(head.val)
	            head = head.next
	        result = dummy = ListNode(0)
	        values[left-1:right] = values[left-1:right][::-1]
	        for v in values:
	            result.next = ListNode(val=v)
	            result = result.next
	        return dummy.next

            	      
			
### 运行结果

	Runtime: 20 ms, faster than 66.00% of Python online submissions for Reverse Linked List II.
	Memory Usage: 13.7 MB, less than 35.28% of Python online submissions for Reverse Linked List II.

### 解析

当然了还可以一次遍历就完成题意任务的要求，思路比较简单，就是在遍历链表的过程中，根据 left 和 right 将链表截取成三个部分， [left，right]  部分属于第二部分，然后在遍历 [left，right] 范围内的节点的同时对这个范围内的节点进行反正，当遍历到 right 结尾的时候，只需要将第一部分拼接反转成功的第二部分，第二部分拼接剩下的第三部分即可得到最终的链表结果。

### 解答

	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def reverseBetween(self, head, left, right):
	        """
	        :type head: ListNode
	        :type left: int
	        :type right: int
	        :rtype: ListNode
	        """
	        if not head or not head.next: return head
	        result = dummy = ListNode(-1000)
	        dummy.next = head
	        for _ in range(left-1):
	            dummy = dummy.next
	        endOfFirst = dummy
	        dummy = dummy.next
	        startOfSecond = dummy
	        last = None
	        for _ in range(left, right+1):
	            nxt = dummy.next
	            dummy.next = last
	            last = dummy
	            dummy = nxt
	        endOfFirst.next = last
	        startOfSecond.next = dummy
	        return result.next
	        
### 运行结果

	Runtime: 24 ms, faster than 36.42% of Python online submissions for Reverse Linked List II.
	Memory Usage: 13.7 MB, less than 64.15% of Python online submissions for Reverse Linked List II.

原题链接：https://leetcode.com/problems/reverse-linked-list-ii/



您的支持是我最大的动力
