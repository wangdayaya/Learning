### 描述


Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return null.

For example, the following two linked lists begin to intersect at node c1:

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/55557a2090d64f08be1d1d74f36767a2~tplv-k3u1fbpfcp-zoom-1.image)

The test cases are generated such that there are no cycles anywhere in the entire linked structure.

Note that the linked lists must retain their original structure after the function returns.

Custom Judge:

The inputs to the judge are given as follows (your program is not given these inputs):

* intersectVal - The value of the node where the intersection occurs. This is 0 if there is no intersected node.
* listA - The first linked list.
* listB - The second linked list.
* skipA - The number of nodes to skip ahead in listA (starting from the head) to get to the intersected node.
* skipB - The number of nodes to skip ahead in listB (starting from the head) to get to the intersected node.


The judge will then create the linked structure based on these inputs and pass the two heads, headA and headB to your program. If you correctly return the intersected node, then your solution will be accepted.


Follow up: Could you write a solution that runs in O(n) time and use only O(1) memory?

Example 1:

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6fd9f1801ba34b23b4a1cb72c75ada7e~tplv-k3u1fbpfcp-zoom-1.image)

	Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
	Output: Intersected at '8'
	Explanation: The intersected node's value is 8 (note that this must not be 0 if the two lists intersect).
	From the head of A, it reads as [4,1,8,4,5]. From the head of B, it reads as [5,6,1,8,4,5]. There are 2 nodes before the intersected node in A; There are 3 nodes before the intersected node in B.
	
	
Example 2:

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2bb686d505a348f38d58bd4aa4aa4ed2~tplv-k3u1fbpfcp-zoom-1.image)

	Input: intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
	Output: Intersected at '2'
	Explanation: The intersected node's value is 2 (note that this must not be 0 if the two lists intersect).
	From the head of A, it reads as [1,9,1,2,4]. From the head of B, it reads as [3,2,4]. There are 3 nodes before the intersected node in A; There are 1 node before the intersected node in B.

Example 3:

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f854a538f49a489988389d8fcede778f~tplv-k3u1fbpfcp-zoom-1.image)

	Input: intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
	Output: No intersection
	Explanation: From the head of A, it reads as [2,6,4]. From the head of B, it reads as [1,5]. Since the two lists do not intersect, intersectVal must be 0, while skipA and skipB can be arbitrary values.
	Explanation: The two lists do not intersect, so return null.



Note:

	The number of nodes of listA is in the m.
	The number of nodes of listB is in the n.
	0 <= m, n <= 3 * 10^4
	1 <= Node.val <= 10^5
	0 <= skipA <= m
	0 <= skipB <= n
	intersectVal is 0 if listA and listB do not intersect.
	intersectVal == listA[skipA] == listB[skipB] if listA and listB intersect.
	 

### 解析

根据题意，就是给出了两个链表的头节点 headA 和 headB ，如果这两个链表相交让我们找出相交的节点，如果不相交直接返回 None 就行。题目给我们还增加了难度，要求我们在 O(1) 级别的内存下完成题目要求。

思路比较简单，最简单的办法就是遍历两个链表 headA 和 headB 计算他们的长度差 d ，让较长的那个先往前遍历 d 个节点，然后两个链表同时往后便利，如果有相交的节点一定会指针相等，遍历结束没有找到说明没有相交。

### 解答
					
	class ListNode(object):
	    def __init__(self, x):
	        self.val = x
	        self.next = None
	
	class Solution(object):
	    def getIntersectionNode(self, headA, headB):
	        """
	        :type head1, head1: ListNode
	        :rtype: ListNode
	        """
	        if not headA or not headB: return None
	        len_a = self.getLen(headA)
	        len_b = self.getLen(headB)
	        diff = abs(len_a-len_b)
	        for i in range(diff):
	            if len_a>len_b:
	                headA = headA.next
	            else:
	                headB = headB.next
	        while headA and headB:
	            if headA == headB:
	                return headA
	            headA = headA.next
	            headB = headB.next
	        return None
	        
	    def getLen(self, L):
	        result = 0
	        while L:
	            result += 1
	            L = L.next
	        return result
            	      
			
### 运行结果

	
	Runtime: 212 ms, faster than 46.07% of Python online submissions for Intersection of Two Linked Lists.
	Memory Usage: 43.3 MB, less than 58.63% of Python online submissions for Intersection of Two Linked Lists.



### 解析



还有一种解法是看了 leetcode 的论坛中的大神写出来的，及其巧妙。

* 用指针 a 指向 headA ，用指针 b 指向 headB 
* 当 a 不等于 b 的时候，一直进行 while 循环，两个链表同时向后遍历，如果 a 不为空则继续下后遍历一个节点，如果 a 为空则让 a 指针指向 headB ；如果 b 不为空则继续下后遍历一个节点，如果 b 为空则让 b 指针指向 headA
* 因为第一次遍历消除了两个链表之间的长度差，所以如果两者相交，在第二次遍历的时候肯定能找到 a == b 的节点，如果两者不相交，则 a 和 b 的遍历结果肯定都为 None 


### 解答
					
	class ListNode(object):
	    def __init__(self, x):
	        self.val = x
	        self.next = None
	
	class Solution(object):
	    def getIntersectionNode(self, headA, headB):
	        """
	        :type head1, head1: ListNode
	        :rtype: ListNode
	        """
	        if not headA or not headB: return None
	        a = headA
	        b = headB
	        while a != b:
	            if not a :
	                a = headB
	            else:
	                a = a.next
	            if not b:
	                b = headA
	            else:
	                b = b.next
	        return a
        	      
			
### 运行结果

	Runtime: 196 ms, faster than 75.21% of Python online submissions for Intersection of Two Linked Lists.
	Memory Usage: 43.4 MB, less than 39.90% of Python online submissions for Intersection of Two Linked Lists.


原题链接：https://leetcode.com/problems/intersection-of-two-linked-lists/


您的支持是我最大的动力
