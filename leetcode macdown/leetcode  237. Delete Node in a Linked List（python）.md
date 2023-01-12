leetcode  237. Delete Node in a Linked List（python）




### 描述

There is a singly-linked list head and we want to delete a node node in it.You are given the node to be deleted node. You will not be given access to the first node of head.All the values of the linked list are unique, and it is guaranteed that the given node node is not the last node in the linked list.

Delete the given node. Note that by deleting the node, we do not mean removing it from memory. We mean:

* The value of the given node should not exist in the linked list.
* The number of nodes in the linked list should decrease by one.
* All the values before node should be in the same order.
* All the values after node should be in the same order.





Example 1:

![](https://assets.leetcode.com/uploads/2020/09/01/node1.jpg)

	Input: head = [4,5,1,9], node = 5
	Output: [4,1,9]
	Explanation: You are given the second node with value 5, the linked list should become 4 -> 1 -> 9 after calling your function.

	
Example 2:

![](https://assets.leetcode.com/uploads/2020/09/01/node2.jpg)

	Input: head = [4,5,1,9], node = 1
	Output: [4,5,9]
	Explanation: You are given the third node with value 1, the linked list should become 4 -> 5 -> 9 after calling your function.



Note:

	The number of the nodes in the given list is in the range [2, 1000].
	-1000 <= Node.val <= 1000
	The value of each node in the list is unique.
	The node to be deleted is in the list and is not a tail node.


### 解析

根据题意，有一个单链表头 head ，我们想要删除其中的节点 node 。给定要删除的节点 node ，并且无法访问 head 的第一个节点。链表的所有值都是唯一的，保证给定的节点 node 不是链表中的最后一个节点。要求删除给定节点。需要注意的是删除节点并不意味着将其从内存中删除。也就是说：

* 给定节点的值不应存在于链表中
* 链表中的节点数应减少 1
* 节点之前的所有值应按相同的顺序排列
* 节点后的所有值应按相同的顺序排列

其实这道题很简单，目标节点已经给出来了，整体思路就是把当前节点的值更新为下一个节点的值，然后将当前节点的下一个节点更新为下一个节点的下一个节点，换句话说就是把自己整容成儿子，再假装儿子养活孙子。

时间复杂度为 O(1) ，空间复杂度为 O(1) 。
### 解答

	class Solution(object):
	    def deleteNode(self, node):
	        """
	        :type node: ListNode
	        :rtype: void Do not return anything, modify node in-place instead.
	        """
	        node.val = node.next.val
	        node.next = node.next.next

### 运行结果

	Runtime: 37 ms, faster than 70.83% of Python online submissions for Delete Node in a Linked List.
	Memory Usage: 13.7 MB, less than 81.20% of Python online submissions for Delete Node in a Linked List.

### 原题链接

https://leetcode.com/problems/delete-node-in-a-linked-list/


您的支持是我最大的动力
