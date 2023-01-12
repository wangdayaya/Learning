leetcode  82. Remove Duplicates from Sorted List II（python）




### 描述


Given the head of a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list. Return the linked list sorted as well.


Example 1:


![](https://assets.leetcode.com/uploads/2021/01/04/linkedlist1.jpg)


	Input: head = [1,2,3,3,4,4,5]
	Output: [1,2,5]
	



Note:

	The number of nodes in the list is in the range [0, 300].
	-100 <= Node.val <= 100
	The list is guaranteed to be sorted in ascending order.



### 解析


根据题意，给定一个排序链表的头部，只要某些节点带有同样的值就都删除，只留下原始链表中只出现过一次的数字。 返回排序好的链表。而且看限制条件，题目已经保证了给出了的链表的值是按照升序排列的。

题目很简单，解题方法也是多样的。最简单的方法就是先从左到右遍历一次链表，将所有的值都存在一个列表 values 中，然后使用计数器对 values 进行统计，然后从左到右遍历 values 中的每个值，如果其出现的数量为 1 ，则创建新的节点，将其接到链表之后，遍历 values 结束之后，返回 result.next 之后即可。

时间复杂度为 O(N) ，空间复杂度为 O(N)。

### 解答
				
	class Solution(object):
	    def deleteDuplicates(self, head):
	        values = []
	        while head:
	            values.append(head.val)   
	            head = head.next
	        c = collections.Counter(values)
	        result = d = ListNode(-1)
	        for v in values:
	            if c[v]==1:
	                d.next = ListNode(v)
	                d = d.next
	        return result.next

            	      
			
### 运行结果

	Runtime: 35 ms, faster than 66.28% of Python online submissions for Remove Duplicates from Sorted List II.
	Memory Usage: 13.3 MB, less than 85.32% of Python online submissions for Remove Duplicates from Sorted List II.

### 解析

 另外的解法就是我们直接在链表上进行操作，这种操作尽管有点复杂， 但是可以将空间复杂度降低为 O(1) ，因为没有开辟新的空间。
 
 我们先定一一个初始化的结果指针 result ，和一个假指针 d ，两个指针的下一个节点都是 head 。我们不断移动链表指针 head 来遍历所有的节点，如果在有下一个节点的情况下，当前节点和下一个节点的值相等，我们就使用 while 循环，不断往后寻找不同值的节点，将其赋值为 d.next ，如果当前节点和下一个节点的值不同，则直接将 d.next 赋值给 d ，遍历完所有的节点，我们返回 result.next 即可。

 时间复杂度为 O(N) ，空间复杂度为O(1) 。

### 解答

	class Solution(object):
	    def deleteDuplicates(self, head):
	        result = d = ListNode(-1, head)
	        while head:
	            if head.next and  head.val==head.next.val:
	                while head.next and head.val == head.next.val:
	                    head = head.next
	                d.next = head.next
	            else:
	                d = d.next
	            head = head.next
	        return result.next
	                

### 运行结果

	Runtime: 23 ms, faster than 98.51% of Python online submissions for Remove Duplicates from Sorted List II.
	Memory Usage: 13.3 MB, less than 97.59% of Python online submissions for Remove Duplicates from Sorted List II.

### 原题链接


https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/


您的支持是我最大的动力
