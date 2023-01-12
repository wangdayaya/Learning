leetcode  2181. Merge Nodes in Between Zeros（python）


### 前言

这是 Weekly Contest 281 的第二题，难度 Medium ，考查的就是对链表的基本操作，难度也不是很难。

### 描述


You are given the head of a linked list, which contains a series of integers separated by 0's. The beginning and end of the linked list will have Node.val == 0.

For every two consecutive 0's, merge all the nodes lying in between them into a single node whose value is the sum of all the merged nodes. The modified list should not contain any 0's.

Return the head of the modified linked list.


Example 1:


![](https://assets.leetcode.com/uploads/2022/02/02/ex1-1.png)

	Input: head = [0,3,1,0,4,5,2,0]
	Output: [4,11]
	Explanation: 
	The above figure represents the given linked list. The modified list contains
	- The sum of the nodes marked in green: 3 + 1 = 4.
	- The sum of the nodes marked in red: 4 + 5 + 2 = 11.
	




Note:

	The number of nodes in the list is in the range [3, 2 * 10^5].
	0 <= Node.val <= 1000
	There are no two consecutive nodes with Node.val == 0.
	The beginning and end of the linked list have Node.val == 0.


### 解析


根据题意，您将获得一个链表的 head ，其中包含一系列由 0 分隔的整数。 链表的开头和结尾的 Node.val == 0。对于每两个连续的 0，将位于它们之间的所有节点合并为一个节点，其值是所有合并节点的总和。修改后的链表不应包含任何值为 0 的节点。返回修改后的链表的 head 。

这个题我在比赛的时候用的是比较投机取巧的方法，它考察的肯定是对链表的基本操作，但是我直接遍历了链表的所有节点，将所有的值都存到了一个列表 L 中，然后通过对列表 L 的操作，将两个 0 之间的值求和并用它初始化为一个节点加入到新的链表 result 中，操作完成之后我们将链表 result.next 返回即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。从耗时上面看刚好能通过，很多时间都消耗在了初始化新的节点操作上了，好险能通过。

### 解答
				

	class Solution(object):
	    def mergeNodes(self, head):
	        """
	        :type head: Optional[ListNode]
	        :rtype: Optional[ListNode]
	        """
	        L = []
	        while head:
	            L.append(head.val)
	            head = head.next
	        result = dummy = ListNode(-1)
	        total = 0
	        for c in L:
	            if c == 0:
	                if total!=0:
	                    dummy.next = ListNode(total)
	                    dummy = dummy.next
	                    total = 0
	            total += c
	        return result.next
            	      
			
### 运行结果

	39 / 39 test cases passed.
	Status: Accepted
	Runtime: 6346 ms
	Memory Usage: 201.7 MB

### 解析

以我的脾气，比赛时候虽然能够为了争取时间而投机取巧节省时间，但是这种无脑的解法没什么技术含量，人家考察链表的操作，我们就应该用链表的操作来完成，而不是借用列表来完成题目。

其实思路也很简单，因为第一个节点就是 0 所以我们使用指针 cur 从第二个节点开始遍历整个链表的节点值，我们还初始化了一个假指针 dummy ，我们使用它将第 i 个非零区间的和放置在链表的第 i 个节点位置上，这样经过遍历到最后，我们只需要将 dummy.next 赋值为 None ，然后返回原链表 head 即可，此时的 head 中的每个值就是每个非零区间的和。


时间复杂度仍然是 O(N)，但是空间复杂度是 O(1)， 因为没有使用额外开辟的空间，只是单纯在原链表上进行了修改。
### 解答

	class Solution(object):
	    def mergeNodes(self, head):
	        """
	        :type head: Optional[ListNode]
	        :rtype: Optional[ListNode]
	        """
	        cur = head.next
	        dummy = head
	        while cur:
	            if cur.val!=0:
	                dummy.val += cur.val
	                cur = cur.next
	            else:
	                if not cur.next:
	                    dummy.next = None
	                    break
	                dummy = dummy.next
	                dummy.val = 0
	                cur = cur.next
	        return head

### 运行结果

	39 / 39 test cases passed.
	Status: Accepted
	Runtime: 4258 ms
	Memory Usage: 132 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-281/problems/merge-nodes-in-between-zeros/


您的支持是我最大的动力
