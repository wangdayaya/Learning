leetcode  142. Linked List Cycle II（python）

### 每日经典

《》 ——（）


### 描述


Given the head of a linked list, return the node where the cycle begins. If there is no cycle, return null.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to (0-indexed). It is -1 if there is no cycle. Note that pos is not passed as a parameter.

Follow up: Can you solve it using O(1) (i.e. constant) memory?

Example 1:

![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist.png)

	Input: head = [3,2,0,-4], pos = 1
	Output: tail connects to node index 1
	Explanation: There is a cycle in the linked list, where tail connects to the second node.
Example 2:

![https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist_test2.png](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist_test2.png)

	Input: head = [1,2], pos = 0
	Output: tail connects to node index 0
	Explanation: There is a cycle in the linked list, where tail connects to the first node.

Example 3:

![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist_test3.png)

	Input: head = [1], pos = -1
	Output: no cycle
	Explanation: There is no cycle in the linked list.

Note:

	The number of the nodes in the list is in the range [0, 10^4].
	-10^5 <= Node.val <= 10^5
	pos is -1 or a valid index in the linked-list.


### 解析


根据题意，给定链表的头节点 head ，返回循环开始的节点。 如果没有循环，则返回 null 。

如果链表中有某个节点可以通过不断跟随 next 指针再次到达，则链表中存在一个循环。 在内部，pos 用于表示开始链表尾部节点的 next 所指向的节点的索引（0-indexed），其实也就是循环点。如果没有循环，则为 -1 。 需要注意的是 pos 不作为参数传递，而且题目要求不能修改链表。题目还给有能力的同学提出了新的要求，能否使用 O(1) 的内存解题。

这道题其实就是在考察怎么定位链表中环入口的位置，我们先用最普通的方法去解决这道题，用我们最朴素的想法，肯定是先遍历一遍链表，将遍历到的某个节点存放入一个集合 s 中，如果到达某个节点已经出现在了集合 s 中，那么说明该节点就是环的入口，如上面的例子一，当遍历链表第二次到达第二个节点的时候，集合中已经有了，所以说明这个节点就是环的入口。这种解法的时间复杂度就是 O(n) ，但是空间复杂度是 O(n) ，尽管是可以通过的，但是没有达到题目使用 O(1) 的内存解题高要求。

其实这也说明了一个最简单的道理，越是简单暴力的算法可能消耗的资源就越多，越是精巧的算法消耗的资源就会越少，当然这也不是绝对的，只是相对的。

### 解答
				

	class Solution(object):
	    def detectCycle(self, head):
	        """
	        :type head: ListNode
	        :rtype: ListNode
	        """
	        s = set()
	        while head:
	            if head not in s :
	                s.add(head)
	            else:
	                return head
	            head = head.next
	        return None
            	      
			
### 运行结果


	Runtime: 72 ms, faster than 16.78% of Python online submissions for Linked List Cycle II.
	Memory Usage: 20.1 MB, less than 16.53% of Python online submissions for Linked List Cycle II.
	
### 解析

其实消耗空间比较多就是为了找第二次出现的点，其实我们完全可以省去这部分的空间消耗，通过快慢指针的遍历来判断是否有环，但这仅仅是第一步，如果没有环直接返回，如果有环才开始后面的步骤。

假设快慢两个指针相遇，快指针慢指针都走过了 m 个节点进入环的入口，快指针走了 a 圈又多走了 b 步，慢指针走了 c 圈又多走了 b 步，因为快指针经过的路程是慢指针的两倍，环的长度为 n ，所以关系为 
	
	 m + a*n + b = 2(m + c*n + b) 
	
化简之后为：

	(a-2c)*n = m+b
	
因为慢指针目前已经比整数圈多走了 b 步，结合上面的公式，我们可以发现如果慢指针再走 m 步，又回凑成整数圈，也就是到达环的入口，此时的 m 也就是让链表的 head 与慢指针同时继续往后走，碰头的地方就是入口。
	
还有就是注意边界条件，当 head 为空的时候直接返回即可，TMD 每次提醒别人注意边界条件，自己每次都被边界条件卡到，真的是讽刺。


### 解答

	class Solution(object):
	    def detectCycle(self, head):
	        """
	        :type head: ListNode
	        :rtype: ListNode
	        """
	        if not head: return 
	        slow = head
	        fast = head
	        tmp = None
	        while fast.next and fast.next.next:
	            slow = slow.next
	            fast = fast.next.next
	            if slow == fast:
	                tmp = slow
	                break
	        if not tmp: return 
	        while head != slow:
	            head = head.next
	            slow = slow.next
	        return slow

### 运行结果

	Runtime: 81 ms, faster than 10.74% of Python online submissions for Linked List Cycle II.
	Memory Usage: 19.5 MB, less than 95.62% of Python online submissions for Linked List Cycle II.

原题链接：https://leetcode.com/problems/linked-list-cycle-ii/



您的支持是我最大的动力
