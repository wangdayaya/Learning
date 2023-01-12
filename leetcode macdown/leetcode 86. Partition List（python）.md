leetcode 86. Partition List （python）




### 描述

Given the head of a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x. You should preserve the original relative order of the nodes in each of the two partitions.



Example 1:

![](https://assets.leetcode.com/uploads/2021/01/04/partition.jpg)

	Input: head = [1,4,3,2,5,2], x = 3
	Output: [1,2,2,4,3,5]

	
Example 2:

	Input: head = [2,1], x = 2
	Output: [1,2]





Note:

	The number of nodes in the list is in the range [0, 200].
	-100 <= Node.val <= 100
	-200 <= x <= 200


### 解析

根据题意，给定链表的 head 和值 x ，使得所有小于 x 的节点都在大于或等于 x 的节点之前，并且保留节点的原始相对顺序。

这道题我们用双指针进行解决即可，我们定义两个头节点 a 和 b ，让 a 去找去 x 小的节点并且连接起来，让 b 去找大于等于 x 的节点并且连接起来，最后只需要将 b 连接到a 的末尾后面即可完成题目。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。

### 解答
	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def partition(self, head, x):
	        a = a_head = ListNode(-1000)
	        b = b_head = ListNode(-1000)
	        while head:
	            print(head.val)
	            if head.val < x:
	                a.next = head
	                a = a.next
	            else:
	                b.next = head
	                b = b.next
	            head = head.next
	        b.next = None
	        a.next = b_head.next
	        return a_head.next

### 运行结果

	Runtime: 32 ms, faster than 52.93% of Python online submissions for Partition List.
	Memory Usage: 13.3 MB, less than 83.26% of Python online submissions for Partition List.


### 原题链接

	https://leetcode.com/problems/partition-list/


您的支持是我最大的动力
