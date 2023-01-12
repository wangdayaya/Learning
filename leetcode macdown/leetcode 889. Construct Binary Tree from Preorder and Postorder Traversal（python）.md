leetcode  889. Construct Binary Tree from Preorder and Postorder Traversal（python）

### 描述

Given two integer arrays, preorder and postorder where preorder is the preorder traversal of a binary tree of distinct values and postorder is the postorder traversal of the same tree, reconstruct and return the binary tree.

If there exist multiple answers, you can return any of them.



Example 1:


![](https://assets.leetcode.com/uploads/2021/07/24/lc-prepost.jpg)

	Input: preorder = [1,2,4,5,3,6,7], postorder = [4,5,2,6,7,3,1]
	Output: [1,2,3,4,5,6,7]
	
Example 2:

	Input: preorder = [1], postorder = [1]
	Output: [1]






Note:

	1 <= preorder.length <= 30
	1 <= preorder[i] <= preorder.length
	All the values of preorder are unique.
	postorder.length == preorder.length
	1 <= postorder[i] <= postorder.length
	All the values of postorder are unique.
	It is guaranteed that preorder and postorder are the preorder traversal and postorder traversal of the same binary tree.



### 解析

根据题意，给定两个整数数组，preorder 和 postorder ，其中 preorder 是不同值的二叉树的前序遍历， postorder 是同一棵树的后序遍历，重构并返回二叉树。如果存在多个答案，要求可以返回其中任何一个。遇到树的问题直接深度优先遍历 DFS 即可解决各种疑难杂症。

使用一个 preorder 和一个 postorder 重构还原一个二叉树要找到其中的规律，从例子一种我们可以看出来：

*  	根结点是 postorder 的最后一个元素 1 ，将其从 postorder 弹出
*  	右子树的根节点为 3 ，其在 preorder 索引为 i ，右子树的节点包含了 preorder[i:]
*  	左子树的节点包含了  preorder[1:i]

不断递归进行上述过程，即可得到重构的二叉树，，这个题还是相对来说比较简单。

### 解答
				
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def constructFromPrePost(self, preorder, postorder):
	        """
	        :type preorder: List[int]
	        :type postorder: List[int]
	        :rtype: TreeNode
	        """
	        def dfs(pre, post):
	            if not pre:
	                return None
	            if len(pre)==1:
	                return TreeNode(post.pop())
	            node = TreeNode(post.pop())
	            idx = pre.index(post[-1])
	            node.right = dfs(pre[idx:], post)
	            node.left = dfs(pre[1:idx], post)
	            return node
	        return dfs(preorder, postorder)

            	      
			
### 运行结果

	Runtime: 40 ms, faster than 75.96% of Python online submissions for Construct Binary Tree from Preorder and Postorder Traversal.
	Memory Usage: 13.8 MB, less than 5.77% of Python online submissions for Construct Binary Tree from Preorder and Postorder Traversal.

原题链接：https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/



您的支持是我最大的动力
