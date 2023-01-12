leetcode  145. Binary Tree Postorder Traversal（python）

### 描述


Given the root of a binary tree, return the postorder traversal of its nodes' values.




Example 1:

![avatar](https://assets.leetcode.com/uploads/2020/08/28/pre1.jpg)

	Input: root = [1,null,2,3]
	Output: [3,2,1]

	
Example 2:


	Input: root = []
	Output: []

Example 3:


	Input: root = [1]
	Output: [1]
	
Example 4:

![avatar](https://assets.leetcode.com/uploads/2020/08/28/pre3.jpg)

	Input: root = [1,2]
	Output: [2,1]


	
Example 5:


![avatar](https://assets.leetcode.com/uploads/2020/08/28/pre2.jpg)

	Input: root = [1,null,2]
	Output: [2,1]

Note:

	The number of the nodes in the tree is in the range [0, 100].
	-100 <= Node.val <= 100


### 解析

根据题意，就是后序遍历二叉树，这种时候只需要使用递归，按照先左后右再根节点的顺序将值都添加到结果列表中。


### 解答
				

	# Definition for a binary tree node.
	# class TreeNode(object):
	#     def __init__(self, val=0, left=None, right=None):
	#         self.val = val
	#         self.left = left
	#         self.right = right
	class Solution(object):
	    def postorderTraversal(self, root):
	        """
	        :type root: TreeNode
	        :rtype: List[int]
	        """
	        r = []
	        def post(root):
	            if not root:
	                return
	            post(root.left)
	            post(root.right)
	            r.append(root.val)
	        post(root)
	        return r
            	      
			
### 运行结果

	Runtime: 12 ms, faster than 94.08% of Python online submissions for Binary Tree Postorder Traversal.
	Memory Usage: 13.3 MB, less than 95.46% of Python online submissions for Binary Tree Postorder Traversal.


原题链接：https://leetcode.com/problems/binary-tree-postorder-traversal/



您的支持是我最大的动力
