leetcode  94. Binary Tree Inorder Traversal（python）

### 描述

Given the root of a binary tree, return the inorder traversal of its nodes' values.





Example 1:

![avatar](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

	Input: root = [1,null,2,3]
	Output: [1,3,2]
	
Example 2:


	Input: root = []
	Output: []

Example 3:

	Input: root = [1]
	Output: [1]

	
Example 4:

![avatar](https://assets.leetcode.com/uploads/2020/09/15/inorder_5.jpg)

	Input: root = [1,2]
	Output: [2,1]
	
Example 5:

![avatar](https://assets.leetcode.com/uploads/2020/09/15/inorder_4.jpg)

	Input: root = [1,null,2]
	Output: [1,2]


Note:


	The number of nodes in the tree is in the range [0, 100].
	-100 <= Node.val <= 100

### 解析


根据题意，就是中序遍历二叉树，这种时候只需要使用递归，按照先左后根再右的顺序将值都添加到结果列表中。

### 解答
				
	# Definition for a binary tree node.
	# class TreeNode(object):
	#     def __init__(self, val=0, left=None, right=None):
	#         self.val = val
	#         self.left = left
	#         self.right = right
	class Solution(object):
	    def inorderTraversal(self, root):
	        """
	        :type root: TreeNode
	        :rtype: List[int]
	        """
	        r = []
	        def inO(root):
	            if not root:
	                return 
	            inO(root.left)
	            r.append(root.val)
	            inO(root.right)
	        inO(root)
	        return r

            	      
			
### 运行结果

	Runtime: 12 ms, faster than 95.55% of Python online submissions for Binary Tree Inorder Traversal.
	Memory Usage: 13.5 MB, less than 16.13% of Python online submissions for Binary Tree Inorder Traversal.


原题链接：https://leetcode.com/problems/binary-tree-inorder-traversal/



您的支持是我最大的动力
