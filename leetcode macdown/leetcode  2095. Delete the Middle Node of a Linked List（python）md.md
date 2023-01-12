leetcode  2095. Delete the Middle Node of a Linked List（python）




### 描述



You are given the head of a linked list. Delete the middle node, and return the head of the modified linked list. The middle node of a linked list of size n is the ⌊n / 2⌋th node from the start using 0-based indexing, where ⌊x⌋ denotes the largest integer less than or equal to x. 

* For n = 1, 2, 3, 4, and 5, the middle nodes are 0, 1, 1, 2, and 2, respectively.


Example 1:

![](https://assets.leetcode.com/uploads/2021/11/16/eg1drawio.png)

	Input: head = [1,3,4,7,1,2,6]
	Output: [1,3,4,1,2,6]
	Explanation:
	The above figure represents the given linked list. The indices of the nodes are written below.
	Since n = 7, node 3 with value 7 is the middle node, which is marked in red.
	We return the new list after removing this node. 

	
Example 2:

![](https://assets.leetcode.com/uploads/2021/11/16/eg2drawio.png)

	Input: head = [1,2,3,4]
	Output: [1,2,4]
	Explanation:
	The above figure represents the given linked list.
	For n = 4, node 2 with value 3 is the middle node, which is marked in red.


Example 3:


![](https://assets.leetcode.com/uploads/2021/11/16/eg3drawio.png)

	Input: head = [2,1]
	Output: [2]
	Explanation:
	The above figure represents the given linked list.
	For n = 2, node 1 with value 1 is the middle node, which is marked in red.
	Node 0 with value 2 is the only node remaining after removing node 1.


Note:

	The number of nodes in the list is in the range [1, 10^5].
	1 <= Node.val <= 10^5


### 解析

根据题意，您将获得链接列表的头指针 head ，删除其中间的节点，并返回修改后的链接列表的头部。大小为 n 的链表从 0 还是索引，中间节点的索引是第 ⌊n / 2⌋ 个节点，其中 ⌊x⌋ 表示小于或等于 x 的最大整数。

* 对于 n = 1、2、3、4 和 5 ，中间节点索引分别为 0、1、1、2 和 2。

其实这道题的关键点有两个：

* 第一个是找到链表的中间节点
* 第二个就是将去掉中间节点的前后相邻节点拼接起来

对于第一点我们常用到的方法就是快慢双指针法，使用一个快指针 quick 每次前进两个节点，使用一个慢指针 slow 每次前进一个节点，当快指针无法前进的时候，慢指针刚好指向了中间的节点位置，而且这种方法还有一个好处就是不用考虑节点的个数是偶数还是奇数。

对于第二点我们只需要使用一个变量 pre 来保存慢指针每次经过的节点，因为当慢指针指向中间节点的时候， pre 指向的刚好是慢指针前一个相邻的节点，这样就方便将 pre 指向慢指针的下一个节点，完成断开链表的拼接问题。

有一种特殊的情况就是链表只有一个节点的时候，我们只需要特殊判断，直接返回 None 即可。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。

### 解答

	class Solution(object):
	    def deleteMiddle(self, head):
	        """
	        :type head: Optional[ListNode]
	        :rtype: Optional[ListNode]
	        """
	        if not head.next:
	            return None
	        slow, quick, pre = head, head, None
	        while quick and quick.next:
	            quick = quick.next.next
	            pre = slow
	            slow = slow.next
	        pre.next = slow.next
	        return head
	


### 运行结果

	Runtime: 5212 ms, faster than 12.68% of Python online submissions for Delete the Middle Node of a Linked List.
	Memory Usage: 96.9 MB, less than 95.39% of Python online submissions for Delete the Middle Node of a Linked List.


### 原题链接

https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/


您的支持是我最大的动力
