leetcode  19. Remove Nth Node From End of List（python）




### 描述


Given the head of a linked list, remove the nth node from the end of the list and return its head.

![](https://assets.leetcode.com/uploads/2020/10/03/remove_ex1.jpg)

Example 1:

	Input: head = [1,2,3,4,5], n = 2
	Output: [1,2,3,5]

	
Example 2:


	Input: head = [1], n = 1
	Output: []

Example 3:


	Input: head = [1,2], n = 1
	Output: [1]


Note:

	The number of nodes in the list is sz.
	1 <= sz <= 30
	0 <= Node.val <= 100
	1 <= n <= sz


### 解析

根据题意，给定链表的头，从链表的末尾删除第 n 个节点并返回它的头。

其实最朴素的方法就是我们直接从头到尾遍历一遍链表，记录链表节点的总数，然后再从头遍历一次链表，将倒数第 n 个节点的前后两个节点连接起来即可。需要注意的是要特别保留多个链表头指针，否则进行一次遍历之后就没有头节点指针了。

时间复杂度为 O(N) ，N 是链表长度，空间复杂度为 O(1) 。

### 解答

	class Solution(object):
	    def removeNthFromEnd(self, head, n):
	        """
	        :type head: ListNode
	        :type n: int
	        :rtype: ListNode
	        """
	        N = 0
	        result = dummy = ListNode(-1, head)
	        while head:
	            N += 1
	            head = head.next
	        N -= n
	        for i in range(1, N+1):
	            dummy = dummy.next
	        dummy.next = dummy.next.next
	        return result.next

### 运行结果

	Runtime: 14 ms, faster than 98.36% of Python online submissions for Remove Nth Node From End of List.
	Memory Usage: 13.4 MB, less than 67.26% of Python online submissions for Remove Nth Node From End of List.


### 解析

其实我们还可以使用栈来解决这个问题，每个节点当作元素压入栈中，当从栈顶弹出第 n 个节点是我们将要删除的节点，将当前栈顶的节点的后继节点指针指向第 n+1 个元素即可，这种方法比较直观容易理解。

时间复杂度为 O(N) ，空间复杂度为 O(N) ，N 是链表的长度。


### 解答

	class Solution(object):
	    def removeNthFromEnd(self, head, n):
	        """
	        :type head: ListNode
	        :type n: int
	        :rtype: ListNode
	        """
	        stack = []
	        result = dummy = ListNode(-1, head)
	        while dummy:
	            stack.append(dummy)
	            dummy = dummy.next
	        for _ in range(n):
	            stack.pop()
	        pre = stack[-1]
	        pre.next = pre.next.next
	        return result.next

### 运行结果

	Runtime: 32 ms, faster than 54.46% of Python online submissions for Remove Nth Node From End of List.
	Memory Usage: 13.4 MB, less than 67.26% of Python online submissions for Remove Nth Node From End of List.

### 解析

解决链表问题，我们同样可以使用经典的双指针法，我们使用两个指针，第一个指针先从前往后进行遍历，第二个指针和第一个指针相差 n 个节点的距离，当第一个指针指向末尾的时候，第二个指针刚好指向被删除的节点位置，然后和上面的一样只需要将前后两个节点的指针进行更新并相连即可。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。
 
### 解答

	class Solution(object):
	    def removeNthFromEnd(self, head, n):
	        """
	        :type head: ListNode
	        :type n: int
	        :rtype: ListNode
	        """
	        first = head
	        dummy = ListNode(-1, head)
	        second = dummy
	        for _ in range(n):
	            first = first.next
	        while first:
	            second = second.next
	            first = first.next 
	        second.next = second.next.next
	        return dummy.next

### 运行结果

	Runtime: 35 ms, faster than 43.32% of Python online submissions for Remove Nth Node From End of List.
	Memory Usage: 13.3 MB, less than 67.26% of Python online submissions for Remove Nth Node From End of List.
### 原题链接

https://leetcode.com/problems/remove-nth-node-from-end-of-list/


您的支持是我最大的动力
