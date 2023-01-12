leetcode  203. Remove Linked List Elements（python）

### 描述

Given the head of a linked list and an integer val, remove all the nodes of the linked list that has Node.val == val, and return the new head.





Example 1:

![](https://assets.leetcode.com/uploads/2021/03/06/removelinked-list.jpg)

	Input: head = [1,2,6,3,4,5,6], val = 6
	Output: [1,2,3,4,5]

	
Example 2:

	Input: head = [], val = 1
	Output: []


Example 3:

	Input: head = [7,7,7,7], val = 7
	Output: []

	



Note:

	The number of nodes in the list is in the range [0, 10^4].
	1 <= Node.val <= 50
	0 <= val <= 50


### 解析


根据题意，就是给出了一个链表的头节点 head 和一个整数 val ，题目要求我们移除链表中所有值和 val 相等的节点，并返回处理之后的新的链表的头。这个题就是考察我们对删除链表节点的基本操作，直接新建一个链表头节点 result ，然后遍历链表中的每个节点，如果节点的值和 val 不一样，就新建一个和当前节点一样值的节点连接到 result 后面，遍历结束，最后返回 result.next 即可。



### 解答
				
	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def removeElements(self, head, val):
	        """
	        :type head: ListNode
	        :type val: int
	        :rtype: ListNode
	        """
	        if not head:return head
	        result = ListNode(0, None)
	        tmp = result
	        while head:
	            if head.val!=val:
	                tmp.next = ListNode(head.val, None)
	                tmp = tmp.next
	            head = head.next
	        return result.next

            	      
			
### 运行结果

	
	Runtime: 76 ms, faster than 35.99% of Python online submissions for Remove Linked List Elements.
	Memory Usage: 23.7 MB, less than 6.39% of Python online submissions for Remove Linked List Elements.

### 解析

可以在原链表上直接进行操作，这样直接快速遍历一次即可完成，并且能节省新的内存开销。思路也很简单：

* 先 while 循环找到头节点 head 不为 val 的位置
* 然后将 head 赋值给 result ，让 result 保存头节点
* 当 head 不为空的时候一直执行 while 循环，如果当前节点的下个节点不为空且值为 val ，则将 head.next = head.next.next ，否则则直接 head = head.next 进行之后的节点遍历。
* 循环结束 返回 result 

### 解答

	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def removeElements(self, head, val):
	        """
	        :type head: ListNode
	        :type val: int
	        :rtype: ListNode
	        """
	        if not head:return head
	        while head and head.val==val:
	            head = head.next
	        result = head
	        while head :
	            if head.next and head.next.val == val:
	                head.next = head.next.next
	            else:
	                head = head.next
	        return result

### 运行结果
	
	Runtime: 56 ms, faster than 94.53% of Python online submissions for Remove Linked List Elements.
	Memory Usage: 20.4 MB, less than 13.14% of Python online submissions for Remove Linked List Elements.

原题链接：https://leetcode.com/problems/remove-linked-list-elements/



您的支持是我最大的动力
