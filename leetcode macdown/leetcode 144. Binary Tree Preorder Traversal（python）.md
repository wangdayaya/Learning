leetcode  144. Binary Tree Preorder Traversal（python）

### 描述


Given the root of a binary tree, return the preorder traversal of its nodes' values.




Example 1:

![avatar](https://assets.leetcode.com/uploads/2020/09/15/inorder_1.jpg)

	Input: root = [1,null,2,3]
	Output: [1,2,3]


	
Example 2:

	Input: root = []
	Output: []


Example 3:

	Input: root = [1]
	Output: [1]

	
Example 4:

![avatar](https://assets.leetcode.com/uploads/2020/09/15/inorder_5.jpg)

	Input: root = [1,2]
	Output: [1,2]
	
Example 5:

![avatar](https://assets.leetcode.com/uploads/2020/09/15/inorder_4.jpg)

	Input: root = [1,null,2]
	Output: [1,2]

Note:


	The number of nodes in the tree is in the range [0, 100].
	-100 <= Node.val <= 100

### 解析


根据题意，就是前序遍历对二叉树进行遍历。直接使用递归的方法，按照先根后左再右的顺序，将节点的值放入结果列表中。

### 解答
				

	# Definition for a binary tree node.
	# class TreeNode(object):
	#     def __init__(self, val=0, left=None, right=None):
	#         self.val = val
	#         self.left = left
	#         self.right = right
	class Solution(object):
	    def preorderTraversal(self, root):
	        """
	        :type root: TreeNode
	        :rtype: List[int]
	        """
	        r = []
	        def pre(root):
	            if not root:
	                return 
	            r.append(root.val)
	            pre(root.left)
	            pre(root.right)
	        pre(root)
	        return r
            	      
			
### 运行结果

	Runtime: 16 ms, faster than 74.54% of Python online submissions for Binary Tree Preorder Traversal.
	Memory Usage: 13.6 MB, less than 17.41% of Python online submissions for Binary Tree Preorder Traversal.


原题链接：https://leetcode.com/problems/binary-tree-preorder-traversal/



您的支持是我最大的动力
