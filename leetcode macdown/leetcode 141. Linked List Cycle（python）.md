leetcode  141. Linked List Cycle（python）

### 描述

Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.



Follow up: Can you solve it using O(1) (i.e. constant) memory?

Example 1:

![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist.png)

	Input: head = [3,2,0,-4], pos = 1
	Output: true
	Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).

	
Example 2:


![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist_test2.png)

	Input: head = [1,2], pos = 0
	Output: true
	Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.

Example 3:


![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist_test3.png)

	Input: head = [1], pos = -1
	Output: false
	Explanation: There is no cycle in the linked list.


Note:

	The number of the nodes in the list is in the range [0, 10^4].
	-10^5 <= Node.val <= 10^5
	pos is -1 or a valid index in the linked-list.


### 解析

根据题意，给出了一个链表的头 head ，让我们检查这个链表中是否有环存在。

题目还给出了环的定义：如果链表中存在可以通过连续进行的下一个指针再次到达的节点，则链表中存在循环。 变量 pos 用于表示 tail 的 next 指针所连接的节点的索引。 请注意， pos 不作为参数传递。

如果有能力的话 ，题目还给出了更高的要求，让我们使用常量级别的内存解题。

其实这种判断链表上是否有环的解法通常是固定的，就是找一个快指针 fast ，一个慢指针 slow ，让 fast 每次前进两步，让 slow 每次前进一步，如果链表中有环，那么 fast 一定会和 slow 再次相遇。



### 解答
				
	class ListNode(object):
	    def __init__(self, x):
	        self.val = x
	        self.next = None
	
	class Solution(object):
	    def hasCycle(self, head):
	        """
	        :type head: ListNode
	        :rtype: bool
	        """
	        slow = head
	        fast = head
	        while fast!=None and fast.next!=None :
	            slow = slow.next
	            fast = fast.next.next
	            if slow == fast:
	                return True
	        return False
			
### 运行结果

	
	Runtime: 52 ms, faster than 57.87% of Python online submissions for Linked List Cycle.
	Memory Usage: 20.4 MB, less than 83.29% of Python online submissions for Linked List Cycle.

	
原题链接：https://leetcode.com/problems/linked-list-cycle/



您的支持是我最大的动力
