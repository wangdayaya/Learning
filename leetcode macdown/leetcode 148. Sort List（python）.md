leetcode  148. Sort List（python）

### 描述


Given the head of a linked list, return the list after sorting it in ascending order.Can you sort the linked list in O(n logn) time and O(1) memory (i.e. constant space)?


Example 1:

![](https://assets.leetcode.com/uploads/2020/09/14/sort_list_1.jpg)

	Input: head = [4,2,1,3]
	Output: [1,2,3,4]

	
Example 2:


![](https://assets.leetcode.com/uploads/2020/09/14/sort_list_2.jpg)

	Input: head = [-1,5,3,4,0]
	Output: [-1,0,3,4,5]

Example 3:

	Input: head = []
	Output: []


Note:

	The number of nodes in the list is in the range [0, 5 * 10^4].
	-10^5 <= Node.val <= 10^5
 


### 解析

根据题意，就是给出了一个链表的头节点 head ，要求我们对其进行升序排序。题目还要求我们使用 O(n logn) 的时间复杂度和 O(1) 的内存。我尝试用了最简单的暴力插入，空间复杂度还是 O(1)，但是 O(n^2) 时间复杂度，还是超时了。


### 解答
				
	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def sortList(self, head):
	        """
	        :type head: ListNode
	        :rtype: ListNode
	        """
	        if not head or not head.next:return head
	        L = ListNode(val=-1000000)
	        L.next  = cur = head
	        while cur.next:
	            pre = L
	            if cur.next.val >= cur.val:
	                cur = cur.next
	                continue
	            while pre.next.val < cur.next.val:
	                pre = pre.next
	            tmp  = cur.next
	            cur.next = tmp.next
	            tmp.next = pre.next
	            pre.next = tmp
	        return L.next

            	      
			
### 运行结果

	Time Limit Exceeded

### 解析

因为这里用到了内置函数 sorted 对值进行排序，又重新建立新的链表，所以时间上复杂度上是 O(n logn)  但是竟然通过了汗颜。但是空间复杂度是 O(n)。

### 解答

	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def sortList(self, head):
	        """
	        :type head: ListNode
	        :rtype: ListNode
	        """
	        if not head or not head.next:
	            return head
	        node = ListNode(val = 0)
	        result = node
	        nlist = []
	        while head != None:
	            nlist.append(head.val)
	            head = head.next
	        nlist = sorted(nlist)
	        for n in nlist:
	            node.next = ListNode(val = n)
	            node = node.next
	        return result.next
	        
### 运行结果
	Runtime: 284 ms, faster than 84.91% of Python online submissions for Sort List.
	Memory Usage: 63.3 MB, less than 12.52% of Python online submissions for Sort List.
	
### 解析

还可以用归并排序，只要是时间复杂度为 O(n logn)  的其他排序算法都可以。
	
### 解答
	
	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def sortList(self, head):
	        """
	        :type head: ListNode
	        :rtype: ListNode
	        """
	        if not head or not head.next:return head
	        mid = self.getMid(head)
	        another = mid.next
	        mid.next = None
	        return self.merge(self.sortList(head), self.sortList(another))
	    
	    def getMid(self, head):
	        fast = slow = head
	        while fast.next and fast.next.next:
	            fast = fast.next.next
	            slow = slow.next
	        return slow
	    
	    def merge(self, A, B):
	        dummy = cur = ListNode(0)
	        while A and B:
	            if A.val > B.val:
	                cur.next = B
	                B = B.next
	            else:
	                cur.next = A
	                A = A.next
	            cur = cur.next
	        if A: cur.next = A
	        if B: cur.next = B
	        return dummy.next
	        
	        
	        
	 
### 运行结果

	Runtime: 544 ms, faster than 44.25% of Python online submissions for Sort List.
	Memory Usage: 45.6 MB, less than 86.45% of Python online submissions for Sort List.
原题链接：https://leetcode.com/problems/sort-list/



您的支持是我最大的动力
