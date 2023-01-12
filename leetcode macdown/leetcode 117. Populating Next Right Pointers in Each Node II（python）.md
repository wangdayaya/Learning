leetcode  117. Populating Next Right Pointers in Each Node II（python）




### 描述

Given a binary tree

	struct Node {
	  int val;
	  Node *left;
	  Node *right;
	  Node *next;
	}

Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.



Example 1:

![](https://assets.leetcode.com/uploads/2019/02/15/117_sample.png)

	Input: root = [1,2,3,4,5,null,7]
	Output: [1,#,2,3,#,4,5,7,#]
	Explanation: Given the above binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.

	
Example 2:

	Input: root = []
	Output: []







Note:


	The number of nodes in the tree is in the range [0, 6000].
	-100 <= Node.val <= 100

### 解析

根据题意，给定一个二叉树，填充每个 next 指针以指向其同层的右节点。 如果没有下一个右节点，则下一个指针应设置为 NULL。最后让我们返回变化之后的二叉树。

从例子中我们可以直观的看出，只是同层之间的节点连接其右边的节点（如果有的话），这就已经将解题的方法告诉我们了，因为这就是需要 BFS 来进行二叉树的节点的遍历来完成的一项任务。

二叉树的 BFS 遍历常见的一种方法就是使用队列来解决，我们定义一个队列 q  ，然后当队列不为空的时候去循环完成每一层的节点操作，每一层都不断从栈的开头弹出一个节点，取其子节点放入栈尾，同时当队列不为空且其所在位置不是同一层的最后一个，那么就将其 next 指针指向队列中的第一个元素（这个就是它右边紧邻的兄弟节点），不断重复上面的过程，就可以完成 next 的指向问题。最后返回改变之后的二叉树即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答
				

	class Solution(object):
	    def connect(self, root):
	        """
	        :type root: Node
	        :rtype: Node
	        """
	        if not root:
	            return
	        q = []
	        q.append(root)
	        while q:
	            N = len(q)
	            count = 0
	            while count < N:
	                n = q.pop(0)
	                if q and count!=N-1:
	                    n.next = q[0]
	                if n.left:
	                    q.append(n.left)
	                if n.right:
	                    q.append(n.right)
	                count += 1
	        return root
            	      
			
### 运行结果

	Runtime: 48 ms, faster than 46.77% of Python online submissions for Populating Next Right Pointers in Each Node II.
	Memory Usage: 15.9 MB, less than 39.46% of Python online submissions for Populating Next Right Pointers in Each Node II.


### 原题链接


https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/


您的支持是我最大的动力
