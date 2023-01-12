leetcode  589. N-ary Tree Preorder Traversal（python）

### 描述

Given the root of an n-ary tree, return the preorder traversal of its nodes' values.

Nary-Tree input serialization is represented in their level order traversal. Each group of children is separated by the null value (See examples)





Example 1:

![avatar](https://assets.leetcode.com/uploads/2018/10/12/narytreeexample.png)

	Input: root = [1,null,3,2,4,null,5,6]
	Output: [1,3,5,6,2,4]

	
Example 2:


![avatar](https://assets.leetcode.com/uploads/2019/11/08/sample_4_964.png)

	Input: root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
	Output: [1,2,3,6,7,11,14,4,8,12,5,9,13,10]



Note:

	The number of nodes in the tree is in the range [0, 10^4].
	0 <= Node.val <= 10^4
	The height of the n-ary tree is less than or equal to 1000.


### 解析


根据题意，就是对 N 叉树进行前序遍历，只需要进行递归，按照对每个二叉子树进行先根后左再右的顺序进行遍历即可，与二叉树不同的是，这里需要对所有的分叉进行遍历，而不只是左右两个子树。

### 解答
					
	"""
	# Definition for a Node.
	class Node(object):
	    def __init__(self, val=None, children=None):
	        self.val = val
	        self.children = children
	"""
	
	class Solution(object):
	    def preorder(self, root):
	        """
	        :type root: Node
	        :rtype: List[int]
	        """
	        r = []
	        def pre(root):
	            if not root:
	                return
	            r.append(root.val)
	            for child in root.children:
	                pre(child)
	        pre(root)
	        return r
	        

            	      
			
### 运行结果

	
	Runtime: 48 ms, faster than 39.55% of Python online submissions for N-ary Tree Preorder Traversal.
	Memory Usage: 16.4 MB, less than 76.08% of Python online submissions for N-ary Tree Preorder Traversal.

原题链接：https://leetcode.com/problems/n-ary-tree-preorder-traversal/



您的支持是我最大的动力
