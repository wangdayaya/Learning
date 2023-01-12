

### 描述

Given the head of a singly linked list, reverse the list, and return the reversed list.

A linked list can be reversed either iteratively or recursively. Could you implement both?


Example 1:

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/10f222a7f78641eb844af510517aeae6~tplv-k3u1fbpfcp-zoom-1.image)
	
	Input: head = [1,2,3,4,5]
	Output: [5,4,3,2,1]



	
Example 2:

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/683aad2c87174c96bdf96750ce32a6e9~tplv-k3u1fbpfcp-zoom-1.image)

	Input: head = [1,2]
	Output: [2,1]


Example 3:


	Input: head = []
	Output: []



Note:

	The number of nodes in the list is the range [0, 5000].
	-5000 <= Node.val <= 5000


### 解析

根据题意，就是给出了一个链表头节点 head ，题目要求我们将这个链表翻转过来，并且返回反转之后的表链的头节点。

题目还给我们提出了更高的要求，让我们尝试用迭代或者递归的方法来实现算法。

要求暂时别管，最简单最暴力美学的办法就是遍历所有的节点，将值都存下来，然后再新建一个链表，倒序将使用这些节点的值新建节点拼接到新的链表后面。


### 解答
				
	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def reverseList(self, head):
	        """
	        :type head: ListNode
	        :rtype: ListNode
	        """
	        if not head: return head
	        values = []
	        while head:
	            values.append(head.val)
	            head = head.next
	        result = ListNode(0,None)
	        current = result
	        values = values[::-1]
	        for value in values:
	            current.next = ListNode(value, None)
	            current = current.next
	        return result.next

            	      
			
### 运行结果

	Runtime: 32 ms, faster than 31.40% of Python online submissions for Reverse Linked List.
	Memory Usage: 17 MB, less than 19.62% of Python online submissions for Reverse Linked List.

### 解析

也可以不使用列表存储所有节点值，在遍历链表的时候，倒序拼接出新的链表。但是这种解法还是使用了额外的内存空间。


### 解答


	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def reverseList(self, head):
	        """
	        :type head: ListNode
	        :rtype: ListNode
	        """
	        if not head: return head
	        result = None
	        pre = None
	        while head:
	            result = ListNode(head.val, pre)
	            pre = ListNode(head.val, pre)
	            head = head.next
	        return result 
	                

### 运行结果

	Runtime: 36 ms, faster than 21.08% of Python online submissions for Reverse Linked List.
	Memory Usage: 17 MB, less than 21.82% of Python online submissions for Reverse Linked List.

### 解析

真正的高手都是不使用额外的内存空间，直接在原链表上进行遍历，从运行结果来看，很明显比前两种解法的运行速度和内存性能都提升了很多。

### 解答

	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def reverseList(self, head):
	        """
	        :type head: ListNode
	        :rtype: ListNode
	        """
	        result = None
	        while head:
	            tmp = head.next
	            head.next = result
	            result = head
	            head = tmp
	        return result

### 运行结果

	Runtime: 24 ms, faster than 79.57% of Python online submissions for Reverse Linked List.
	Memory Usage: 15.3 MB, less than 79.29% of Python online submissions for Reverse Linked List.
	
原题链接：https://leetcode.com/problems/reverse-linked-list/



您的支持是我最大的动力
