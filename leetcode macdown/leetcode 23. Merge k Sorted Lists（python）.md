leetcode  23. Merge k Sorted Lists（python）




### 描述

You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.



Example 1:

	Input: lists = [[1,4,5],[1,3,4],[2,6]]
	Output: [1,1,2,3,4,4,5,6]
	Explanation: The linked-lists are:
	[
	  1->4->5,
	  1->3->4,
	  2->6
	]
	merging them into one sorted list:
	1->1->2->3->4->4->5->6

	
Example 2:

	Input: lists = []
	Output: []





Note:

	k == lists.length
	0 <= k <= 10^4
	0 <= lists[i].length <= 500
	-10^4 <= lists[i][j] <= 10^4
	lists[i] is sorted in ascending order.
	The sum of lists[i].length won't exceed 10^4.


### 解析

根据题意，给定一个由 k 个链表组成的 lists ，每个链表按升序排序。将所有链表合并为一个排序的链表并返回。这道题虽然标记的难度是 Hard ，但是看了限制条件之后就会发现其实可以用暴力解题，因为 k 最大为 10^4 ，每个 list 的长度最大为 500 ，所以遍历一次时间复杂度为 O(10^6) 不会超时，在 O(N) 之内，然后排序总的时间复杂度为 O(NlogN) ，还是满足题意的不会超时。

我们先用一个列表 nodes 保存左右节点的值，然后遍历所有的链表节点，在得到新的 nodes 之后对其进行升序排序，我们定义一个头节点 point/result ，遍历 nodes 中的每个值，将节点的值转化为 ListNode 类型，然后串联到 point 指针上就可以。遍历结束我们只需要返回 result.next 即可。空间复杂度为 O(N) 。

其实这种解法比较取巧，我们在比赛的时候时间紧迫，如果在限制条件不严苛的情况下是可以这样解答， 但是在平时的训练过程单中还是要使用其他的解法来解题。


### 解答
				

	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def mergeKLists(self, lists):
	        """
	        :type lists: List[ListNode]
	        :rtype: ListNode
	        """
	        nodes = []
	        for l in lists:
	            while l:
	                nodes.append(l.val)
	                l = l.next
	        point = result = ListNode(0)
	        for node in sorted(nodes):
	            point.next = ListNode(node)
	            point = point.next
	        return result.next
	            
	            
            	      
			
### 运行结果


	Runtime: 96 ms, faster than 85.32% of Python online submissions for Merge k Sorted Lists.
	Memory Usage: 22.3 MB, less than 36.77% of Python online submissions for Merge k Sorted Lists.

### 原题链接

https://leetcode.com/problems/merge-k-sorted-lists/



您的支持是我最大的动力
