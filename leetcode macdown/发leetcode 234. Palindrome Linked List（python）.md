leetcode  234. Palindrome Linked List（python）

### 描述

Given the head of a singly linked list, return true if it is a palindrome.



Example 1:

![avatar](https://assets.leetcode.com/uploads/2021/03/03/pal1linked-list.jpg)

	Input: head = [1,2,2,1]
	Output: true
	
Example 2:

![avatar](https://assets.leetcode.com/uploads/2021/03/03/pal2linked-list.jpg)

	Input: head = [1,2]
	Output: false





Note:

	The number of nodes in the list is in the range [1, 10^5].
	0 <= Node.val <= 9


### 解析

根据题意，就是判断一个列表中包含的值，是否是回文序列。简单的方法就是遍历找出链表中的数放到列表 r 中，然后遍历判断 r[i] 和 r[n-i-1] 是否相等，如果不想等直接返回 False ，否则遍历结束直接返回 True 。


### 解答
				

	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def isPalindrome(self, head):
	        """
	        :type head: ListNode
	        :rtype: bool
	        """
	        if not head.next:
	            return True
	        r = [head.val]
	        while head.next:
	            head = head.next
	            r.append(head.val)
	        n = len(r)
	        for i in range(n//2):
	            if r[i]!=r[n-i-1]:
	                return False
	        return True
            	      
			
### 运行结果

	Runtime: 1120 ms, faster than 51.01% of Python online submissions for Palindrome Linked List.
	Memory Usage: 86.2 MB, less than 10.38% of Python online submissions for Palindrome Linked List.

### 解析

其实上面的过程还可以简化一下，只需要将得到的链表中的值组成的列表，直接进行逆向比对即可判断是否为回文链表。

### 解答

	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def isPalindrome(self, head):
	        """
	        :type head: ListNode
	        :rtype: bool
	        """
	        result = []
	        while head:
	            result.append(head.val)
	            head = head.next
	        return result == result[::-1]
	        
### 运行结果

	Runtime: 1048 ms, faster than 68.12% of Python online submissions for Palindrome Linked List.
	Memory Usage: 85.5 MB, less than 17.93% of Python online submissions for Palindrome Linked List.

### 解析
上面两种解法在本质上没有区别，可以看出两者的运行速度都相当慢，因为有一部分时间都消耗在了组织新的列表上了，最省事的解法肯定是一次遍历。思路如下：

* 定位到链表的中间点
* 然后将后半部分的链表节点进行逆序排列成新的链表
* 将新的链表与前半部分进行比较，相等即为回文链表

很明显运行速度和所用内存都有了明显提升。
### 解答


	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def isPalindrome(self, head):
	        """
	        :type head: ListNode
	        :rtype: bool
	        """
	        if not head or not head.next:
	            return True
	        def reverseList(self, head):
	            newhead=None
	            while head:
	                p=head
	                head=head.next
	                p.next=newhead
	                newhead=p
	            return newhead   
	        slow=fast=head
	        while fast and fast.next:
	            fast=fast.next.next
	            slow=slow.next
	        if fast:
	            slow=slow.next
	        newhead=reverseList(self,slow)
	        while newhead:
	            if newhead.val!=head.val:
	                return False
	            newhead=newhead.next
	            head=head.next
	        return True
	    

### 运行结果

	Runtime: 1016 ms, faster than 74.84% of Python online submissions for Palindrome Linked List.
	Memory Usage: 66.9 MB, less than 86.18% of Python online submissions for Palindrome Linked List.

原题链接：https://leetcode.com/problems/palindrome-linked-list/



您的支持是我最大的动力
