leetcode  25. Reverse Nodes in k-Group（python）

### 描述


Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.

k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.

You may not alter the values in the list's nodes, only nodes themselves may be changed.

Follow-up: Can you solve the problem in O(1) extra memory space?


Example 1:

![](https://assets.leetcode.com/uploads/2020/10/03/reverse_ex1.jpg)

	Input: head = [1,2,3,4,5], k = 2
	Output: [2,1,4,3,5]

	
Example 2:

![](https://assets.leetcode.com/uploads/2020/10/03/reverse_ex2.jpg)

	Input: head = [1,2,3,4,5], k = 3
	Output: [3,2,1,4,5]


Example 3:

	Input: head = [1,2,3,4,5], k = 1
	Output: [1,2,3,4,5]

	
Example 4:

	Input: head = [1], k = 1
	Output: [1]

	


Note:

	The number of nodes in the list is in the range sz.
	1 <= sz <= 5000
	0 <= Node.val <= 1000
	1 <= k <= sz


### 解析

根据题意，给定一个链表，从左到右每次将链表的 k 个节点进行反转并返回其修改后的链表。k 是一个正整数，小于或等于链表的长度。 如果节点数不是 k 的倍数，那么最后剩下的节点应该保持原样。注意不能更改列表节点中的值，只能更改节点本身。题目还给有能力的同学提出了更高的要求，那就是是否能用 O(1) 的空间复杂度解决这道题。

当然最简单的办法就是将链表中各个节点中的值遍历一遍存入列表中，然后再对每 k 个元素建立子链表进行反转，最后对所有的子链表进行拼接即可。这样做肯定可以通过，因为数据量很小，但是肯定不是题目考察的本意。


### 解答
				

	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def reverseKGroup(self, head, k):
	        """
	        :type head: ListNode
	        :type k: int
	        :rtype: ListNode
	        """
	        if not head or not head.next: return head
	        values = []
	        while head:
	            tmp = []
	            for _ in range(k):
	                if not head: break
	                tmp.append(head.val)
	                head = head.next
	            values.append(tmp)
	        result = dummy = ListNode(0)
	        for tmp in values:
	            if len(tmp)<k:
	                for value in tmp:
	                    node = ListNode(value)
	                    dummy.next = node
	                    dummy = dummy.next
	            else:
	                for i in range(len(tmp)-1, -1, -1):
	                    node = ListNode(tmp[i])
	                    dummy.next = node
	                    dummy = dummy.next
	        return result.next
	            
	            
            	      
			
### 运行结果

	Runtime: 52 ms, faster than 25.31% of Python online submissions for Reverse Nodes in k-Group.
	Memory Usage: 16.6 MB, less than 5.68% of Python online submissions for Reverse Nodes in k-Group.

### 解析


另外我们可以像题目中要求的那样使用 O(1) 的空间复杂度进行解题，直接在原链表上进行操作。思路也比较简单：

* 开始就是沿着链表从左往右遍历并计数，如果小于 k 则说明已经到了最后的部分而且不需要反转所以直接返回链表的头节点即可
* 否则就使用链表反转的操作将刚经过的 k 个节点进行反转，拼接到原来的指针上
* 循环执行使用上面的两个过程，一直到链表结束，既可以返回结果

从运行结果看出使用的内存空间确实比上一种方法少了很多，速度也有了提升，因为是直接在原链表上进行的操作，没有额外的操作，其实这道题考察的就是链表反转、链表拼接、链表遍历、链表位置记忆等基本操作。
### 解答


	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def reverseKGroup(self, head, k):
	        """
	        :type head: ListNode
	        :type k: int
	        :rtype: ListNode
	        """
	        if not head or not head.next: return head
	        result = ListNode(0)
	        result.next =  head
	        dummy  = result
	        while dummy.next:
	            count = 0
	            group = dummy
	            while dummy.next and count<k:
	                dummy = dummy.next
	                count += 1
	            if count < k: return result.next
	            nxt = dummy.next
	            
	            last = group.next
	            cur = group.next.next
	            for i in range(k-1):
	                future = cur.next
	                cur.next = last
	                last, cur = cur, future
	                
	            dummy = group.next
	            group.next = last
	            dummy.next = nxt
	            
	        return result.next
	            
	                

### 运行结果

	Runtime: 40 ms, faster than 75.43% of Python online submissions for Reverse Nodes in k-Group.
	Memory Usage: 15.2 MB, less than 51.60% of Python online submissions for Reverse Nodes in k-Group.


原题链接：https://leetcode.com/problems/reverse-nodes-in-k-group/



您的支持是我最大的动力
