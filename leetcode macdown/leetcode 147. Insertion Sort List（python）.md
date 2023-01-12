leetcode  147. Insertion Sort List（python）

### 描述

Given the head of a singly linked list, sort the list using insertion sort, and return the sorted list's head.

The steps of the insertion sort algorithm:

* Insertion sort iterates, consuming one input element each repetition and growing a sorted output list.
* At each iteration, insertion sort removes one element from the input data, finds the location it belongs within the sorted list and inserts it there.
* It repeats until no input elements remain.

The following is a graphical example of the insertion sort algorithm. The partially sorted list (black) initially contains only the first element in the list. One element (red) is removed from the input data and inserted in-place into the sorted list with each iteration.



Example 1:


![](https://assets.leetcode.com/uploads/2021/03/04/sort1linked-list.jpg)
	
	Input: head = [4,2,1,3]
	Output: [1,2,3,4]
	
Example 2:

![](https://assets.leetcode.com/uploads/2021/03/04/sort2linked-list.jpg)

	Input: head = [-1,5,3,4,0]
	Output: [-1,0,3,4,5]





Note:

	The number of nodes in the list is in the range [1, 5000].
	-5000 <= Node.val <= 5000


### 解析

根据题意，给定一个单向链表 head ，使用插入排序对链表进行排序，并返回排序后的列表 head 。

插入排序算法的步骤：

* 插入排序进行迭代，每次重复消耗一个输入元素并生成一个排序的输出列表
* 在每次迭代中，插入排序从输入数据中删除一个元素，在排序列表中找到它所属的位置并将其插入到那里
* 重复直到没有输入元素剩余

其实考察的内容很基础，就是对链表节点的搜索和插入，我们每遍历一个新的节点，找到其在已经排好序的链表中应该插入的位置，进行节点的插入操作即可。

### 解答
				

	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def insertionSortList(self, head):
	        """
	        :type head: ListNode
	        :rtype: ListNode
	        """
	        if not head: return head
	        L = ListNode(val=-10000)
	        L.next = curr = head
	        while curr.next:
	            pre = L
	            if curr.next.val>=curr.val:
	                curr = curr.next
	                continue 
	            while pre.next.val<curr.next.val:
	                pre = pre.next
	            tmp = curr.next
	            curr.next = tmp.next
	            tmp.next = pre.next
	            pre.next = tmp
	        return L.next
            	      
			
### 运行结果
	
	Runtime: 172 ms, faster than 81.63% of Python online submissions for Insertion Sort List.
	Memory Usage: 17.2 MB, less than 15.65% of Python online submissions for Insertion Sort List.



原题链接：https://leetcode.com/problems/insertion-sort-list/



您的支持是我最大的动力
