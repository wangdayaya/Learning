leetcode  143. Reorder List（python）

### 描述



You are given the head of a singly linked-list. The list can be represented as:

L<sub>0</sub> → L<sub>1</sub> → … → L<sub>n - 1</sub> → L<sub>n</sub>

Reorder the list to be on the following form:

L<sub>0</sub> → L<sub>n</sub> → L<sub>1</sub> → L<sub>n - 1</sub> → L<sub>2</sub> → L<sub>n - 2</sub> → …

You may not modify the values in the list's nodes. Only nodes themselves may be changed.

Example 1:

![](https://assets.leetcode.com/uploads/2021/03/04/reorder1linked-list.jpg)

	Input: head = [1,2,3,4]
	Output: [1,4,2,3]

	
Example 2:

![](https://assets.leetcode.com/uploads/2021/03/09/reorder2-linked-list.jpg)

	Input: head = [1,2,3,4,5]
	Output: [1,5,2,4,3]







Note:

	The number of nodes in the list is in the range [1, 5 * 10^4].
	1 <= Node.val <= 1000


### 解析


根据题意，给定了一个单向链表的头指针 head 。 该列表可以表示为：

L<sub>0</sub> → L<sub>1</sub> → … → L<sub>n - 1</sub> → L<sub>n</sub>

题目要求我们将链表重新排序为以下形式，不能修改节点中的值，只能更改节点本身：

L<sub>0</sub> → L<sub>n</sub> → L<sub>1</sub> → L<sub>n - 1</sub> → L<sub>2</sub> → L<sub>n - 2</sub> → …

其实看完题意，思路还是比较容易想出来的，将原链表以中间的节点为中心，一分为二，然后将第二个链表进行反转，然后同时从左往右遍历两个链表，每次从第一个链表取一个节点拼接到新的链表，再从第二个链表取一个节点拼接到新的链表。最后得到的链表就是答案。考察的都是链表的基本操作，比如找中间节点，反转链表，拼接链表，遍历链表等等。

### 解答
				
	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def reorderList(self, head):
	        """
	        :type head: ListNode
	        :rtype: None Do not return anything, modify head in-place instead.
	        """
	        if not head or not head.next:return head
	        node = self.getMid(head)
	        head2 = node.next
	        node.next = None
	        head2 = self.getReverse(head2)
	        self.merge(head, head2)
	    
	    def getMid(self, head):
	        fast = head
	        slow = head
	        while fast and fast.next:
	            fast = fast.next.next
	            slow = slow.next
	        return slow
	    
	    def getReverse(self, head):
	        cur = head
	        last = None
	        while cur:
	            nxt = cur.next
	            cur.next = last
	            last = cur
	            cur = nxt
	        return last
	    
	    def merge(self, head1, head2):
	        result = ListNode(0)
	        while head1 or head2:
	            if head1:
	                result.next = head1
	                head1 = head1.next
	                result = result.next
	            if head2:
	                result.next = head2
	                head2 = head2.next
	                result = result.next

            	      
			
### 运行结果

	Runtime: 88 ms, faster than 90.35% of Python online submissions for Reorder List.
	Memory Usage: 31.3 MB, less than 48.45% of Python online submissions for Reorder List.


原题链接：https://leetcode.com/problems/reorder-list/



您的支持是我最大的动力
