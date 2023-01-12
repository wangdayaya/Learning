leetcode  21. Merge Two Sorted Lists（python）

### 描述


Merge two sorted linked lists and return it as a sorted list. The list should be made by splicing together the nodes of the first two lists.


Example 1:

![](https://assets.leetcode.com/uploads/2020/10/03/merge_ex1.jpg)

	Input: l1 = [1,2,4], l2 = [1,3,4]
	Output: [1,1,2,3,4,4]
	
Example 2:


	Input: l1 = [], l2 = []
	Output: []

Example 3:

	Input: l1 = [], l2 = [0]
	Output: [0]


	


Note:

	The number of nodes in both lists is in the range [0, 50].
	-100 <= Node.val <= 100
	Both l1 and l2 are sorted in non-decreasing order.



### 解析

根据题意，就是将给出的连个链表 l1 和 l2 合并成一个有序的链表并返回。所以我们可以根据合并的思路来完成这个题的解法。

* 初始化 result 和 current 为起始节点；
* while 循环当 l1 和 l2 都不为空的时候，然后判断如果 l1.val <= l2.val ，那么就将 l1 赋予 current.next ，并更新 l1 为下一个节点，否则将 l2 赋予 current.next ，并更新 l2 为下一个节点，找出第一个节点之后，更新 current 为 current.next ；
* 一直循环上述步骤直到跳出 while ，表示已经至少有一个链表遍历结束；
* 可能存在一个较长的链表没有遍历结束或者可能两个链表都已经遍历结束，所以将剩下的部分 l1 or l2  赋予 current.next ；
* 最后因为第一个节点是起始节点，所以返回的是 result.next ，也就是返回第二个节点之后的链表


### 解答
				
	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def mergeTwoLists(self, l1, l2):
	        """
	        :type l1: ListNode
	        :type l2: ListNode
	        :rtype: ListNode
	        """
	        result = current = ListNode(0)
	        while l1 and l2:
	            if l1.val <= l2.val:
	                current.next = l1
	                l1 = l1.next
	            else:
	                current.next = l2
	                l2 = l2.next
	            current = current.next 
	        current.next = l1 or l2 
	        return result.next	        

            	      
			
### 运行结果

	Runtime: 20 ms, faster than 93.79% of Python online submissions for Merge Two Sorted Lists.
	Memory Usage: 13.6 MB, less than 11.43% of Python online submissions for Merge Two Sorted Lists.

### 解析

另外，我们还可以用递归的方法，定义函数，只比较当前两个链表的当前节点的大小，后面的链表交给递归函数去完成将两个节点比较并链接的作用。递归出口就是当 l1 或者 l2 遍历结束的时候就返回 l1 or l2 。

### 解答
				

	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def mergeTwoLists(self, l1, l2):
	        """
	        :type l1: ListNode
	        :type l2: ListNode
	        :rtype: ListNode
	        """
	        if not l1 or not l2 :
	            return l1 or l2
	        if l1.val<=l2.val:
	            l1.next = self.mergeTwoLists(l1.next,l2)
	            return l1
	        else:
	            l2.next = self.mergeTwoLists(l1, l2.next)
	            return l2
	            
	            
	        	
			
### 运行结果

	Runtime: 24 ms, faster than 77.38% of Python online submissions for Merge Two Sorted Lists.
	Memory Usage: 13.5 MB, less than 62.13% of Python online submissions for Merge Two Sorted Lists.


原题链接：https://leetcode.com/problems/merge-two-sorted-lists/



您的支持是我最大的动力
