leetcode 700. Search in a Binary Search Tree （python）




### 描述

You are given the root of a binary search tree (BST) and an integer val.

Find the node in the BST that the node's value equals val and return the subtree rooted with that node. If such a node does not exist, return null.



Example 1:


![](https://assets.leetcode.com/uploads/2021/01/12/tree1.jpg)

	Input: root = [4,2,7,1,3], val = 2
	Output: [2,1,3]
	
Example 2:


![](https://assets.leetcode.com/uploads/2021/01/12/tree2.jpg)

	Input: root = [4,2,7,1,3], val = 5
	Output: []




Note:

	The number of nodes in the tree is in the range [1, 5000].
	1 <= Node.val <= 10^7
	root is a binary search tree.
	1 <= val <= 10^7


### 解析

根据题意，给定二叉搜索树 (BST) 的根和一个整数 val 。在 BST 中找到节点值等于 val 的节点，并返回以该节点为根的子树。 如果这样的节点不存在，则返回 null 。

这是一道很简单的题目，考察的就是遍历 BST 寻找目标整数值，最常规的方法就是使用 DFS 来解题，其实树类型的题目大部分题目都适合用 DFS 来解决。

查找这个 val 的过程无非就是每次和一个节点的值进行比对，如果某个节点的值等于 val 就直接返回这个节点，否则就去判断它的左节点和右节点的情况，这样就是一个递归的过程，递归的出口就是当找到 val 直接返回节点或者遍历所有节点之后没有 val 直接返回 None 。思路还是很清晰的，解决速度也很快。

时间复杂度是 O(N)，空间复杂度为 O(N) 。


### 解答
				

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def searchBST(self, root, val):
	        """
	        :type root: TreeNode
	        :type val: int
	        :rtype: TreeNode
	        """
	        if not root: 
	            return None
	        if root.val == val:
	            return root
	        return self.searchBST(root.left, val) or self.searchBST(root.right, val)
	        
            	      
			
### 运行结果


	Runtime: 69 ms, faster than 81.14% of Python online submissions for Search in a Binary Search Tree.
	Memory Usage: 17.3 MB, less than 57.49% of Python online submissions for Search in a Binary Search Tree.

### 原题链接


https://leetcode.com/problems/search-in-a-binary-search-tree/


您的支持是我最大的动力
