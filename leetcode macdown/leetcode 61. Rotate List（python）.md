leetcode  61. Rotate List（python）




### 描述


Given the head of a linked list, rotate the list to the right by k places.


Example 1:

![](https://assets.leetcode.com/uploads/2020/11/13/rotate1.jpg)

	Input: head = [1,2,3,4,5], k = 2
	Output: [4,5,1,2,3]

	
Example 2:

![](https://assets.leetcode.com/uploads/2020/11/13/roate2.jpg)

	Input: head = [0,1,2], k = 4
	Output: [2,0,1]





Note:

* 	The number of nodes in the list is in the range [0, 500].
* 	-100 <= Node.val <= 100
* 	0 <= k <= 2 * 10^9


### 解析

根据题意，给定链表的头指针 head ，返回将链表向右旋转 k 位之后的链表。

题意是很简单的，其实思路也就是根据题意的操作描述进行的：

* 我们先遍历一次链表，找出链表的长度
* 然后将链表的末尾之后再接上 head 
* 经过计算 num=N-k%N  ，我们可以再从 head 往后找 num 个节点，此时其实就是经过变换之后的末尾节点，然后将末尾节点的 next 置为 None 即可
* 最后返回经过变换之后的头节点 result 即可

时间复杂度为 O(N)，空间复杂度为 O(1)。




### 解答
				

	class Solution(object):
	    def rotateRight(self, head, k):
	        if not head or k==0:
	            return head
	        N = 1
	        result = head
	        while result.next:
	            N += 1
	            result = result.next
	        result.next = head
	        num = N - k%N 
	        result = head
	        end = None
	        for _ in range(num):
	            end = result
	            result = result.next
	        end.next = None
	        return result
### 运行结果

	Runtime: 43 ms, faster than 29.14% of Python online submissions for Rotate List.
	Memory Usage: 13.4 MB, less than 84.00% of Python online submissions for Rotate List.


### 解析

其实还有一种比较笨拙的方法，那就是将所有的节点的值都放入一个列表中，然后经过计算得到需要把前 num 个节点放到列表末尾，这种解法比较直接简单，但是空间复杂度会上升到 O(N)，因为新开辟了空间存放所有节点的值。时间复杂度没有变化，还是 O(N) 。

### 解答

	class Solution(object):
	    def rotateRight(self, head, k):
	        if not head or k==0:
	            return head
	        L  = []
	        while head:
	            L.append(head.val)
	            head = head.next
	        result = dummy = ListNode(-1)
	        N = len(L)
	        num = N-k%N 
	        L = L[num:] + L[:num]
	        for v in L:
	            dummy.next = ListNode(v)
	            dummy = dummy.next
	        return result.next
	        
### 运行结果

	Runtime: 35 ms, faster than 51.88% of Python online submissions for Rotate List.
	Memory Usage: 13.5 MB, less than 60.71% of Python online submissions for Rotate List.
### 原题链接


https://leetcode.com/problems/rotate-list/



您的支持是我最大的动力
