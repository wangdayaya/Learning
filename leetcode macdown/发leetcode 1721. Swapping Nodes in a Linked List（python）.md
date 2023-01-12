

### 描述

You are given the head of a linked list, and an integer k.

Return the head of the linked list after swapping the values of the k<sup>th</sup> node from the beginning and the k<sup>th</sup> node from the end (the list is 1-indexed).


Example 1:

![](https://assets.leetcode.com/uploads/2020/09/21/linked1.jpg)

	Input: head = [1,2,3,4,5], k = 2
	Output: [1,4,3,2,5]
	
Example 2:

	Input: head = [7,9,6,6,7,8,3,0,9,5], k = 5
	Output: [7,9,6,6,8,7,3,0,9,5]

Example 3:

	Input: head = [1], k = 1
	Output: [1]
	
Example 4:

	Input: head = [1,2], k = 1
	Output: [2,1]
	
Example 5:

	Input: head = [1,2,3], k = 2
	Output: [1,2,3]

Note:

	The number of nodes in the list is n.
	1 <= k <= n <= 10^5
	0 <= Node.val <= 100


### 解析

根据题意，就是给出了一个链表，并且给出了一个数字，要求我们用将链表的正向第 k 个值和逆向的第 k 个值进行交换，最后返回新的链表的头节点 head 。其实思路比较简单：

* 初始化 first 和 last 都指向 head 
* 因为一开始就是第一个位置，所以遍历 k-1 次，将 first 指向正向的第 k 个节点
* 然后再将 first 赋予一个新的指针 new\_pointer ，让 new_pointer 和 last 同时向后进行节点遍历，直到 new\_pointer.next 为空，此时所找到的 last 就是逆向的第 k 个节点，将两者的值进行交换即可
* 返回新生成的链表头节点 head 

### 解答
				
	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def swapNodes(self, head, k):
	        """
	        :type head: ListNode
	        :type k: int
	        :rtype: ListNode
	        """
	        first = last = head
	        for _ in range(1,k):
	            first = first.next
	        new_pointer = first
	        while new_pointer.next:
	            last = last.next
	            new_pointer = new_pointer.next
	        first.val, last.val = last.val, first.val
	        return head
	        
            	      
			
### 运行结果


	Runtime: 1610 ms, faster than 33.72% of Python online submissions for Swapping Nodes in a Linked List.
	Memory Usage: 92.8 MB, less than 22.87% of Python online submissions for Swapping Nodes in a Linked List.

### 解析

从上面可以启发我们其实关键就在于找正向和逆向的第 k 个位置，然后进行值的交换，这可以利用列表进行操作：

* 用 head 赋给 current ，表示一个指向当前节点的变量
* 然后使用 current 不断向下进行遍历，按顺序将遍历到的每个节点的索引存入列表 nodes 中
* 将 nodes 中索引为 k-1 的节点和索引为 len(nodes)-k 的节点的值进行交换
* 最后返回 head 即可



### 解答
				
	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def swapNodes(self, head, k):
	        """
	        :type head: ListNode
	        :type k: int
	        :rtype: ListNode
	        """
	        nodes = []
	        current = head
	        while current:
	            nodes.append(current)
	            current = current.next
	        nodes[k-1].val, nodes[len(nodes)-k].val = nodes[len(nodes)-k].val, nodes[k-1].val
	        return head
            	      
			
### 运行结果
	
	Runtime: 2043 ms, faster than 20.53% of Python online submissions for Swapping Nodes in a Linked List.
	Memory Usage: 91.3 MB, less than 49.27% of Python online submissions for Swapping Nodes in a Linked List.

原题链接：https://leetcode.com/problems/swapping-nodes-in-a-linked-list/


您的支持是我最大的动力
