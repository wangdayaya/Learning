leetcode  102. Binary Tree Level Order Traversal（python）




### 描述

Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).



Example 1:

![](https://assets.leetcode.com/uploads/2021/02/19/tree1.jpg)

	Input: root = [3,9,20,null,null,15,7]
	Output: [[3],[9,20],[15,7]]

	
Example 2:

	Input: root = [1]
	Output: [[1]]


Example 3:

	Input: root = []
	Output: []



Note:

	The number of nodes in the tree is in the range [0, 2000].
	-1000 <= Node.val <= 1000


### 解析

根据题意，给定二叉树的根 root ，按照从上到下返回每一层的节点值列表。 

其实这道题考察的就是使用 BFS 来进行二叉树的层序遍历，我们最常见的做法是用队列来模拟整个过程。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def levelOrder(self, root):
	        """
	        :type root: TreeNode
	        :rtype: List[List[int]]
	        """
	        if not root: return []
	        q  = [root]
	        result = []
	        while q:
	            L = len(q)
	            level = []
	            for i in range(L):
	                node = q.pop(0)
	                level.append(node.val)
	                if node.left:
	                    q.append(node.left)
	                if node.right:
	                    q.append(node.right)
	            result.append(level)
	        return result
### 运行结果

	Runtime: 44 ms, faster than 16.25% of Python online submissions for Binary Tree Level Order Traversal.
	Memory Usage: 13.5 MB, less than 94.70% of Python online submissions for Binary Tree Level Order Traversal.



### 原题链接

https://leetcode.com/problems/binary-tree-level-order-traversal/


您的支持是我最大的动力
