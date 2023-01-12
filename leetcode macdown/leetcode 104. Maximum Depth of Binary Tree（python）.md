leetcode 104. Maximum Depth of Binary Tree （python）

### 描述

Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.



Example 1:

![](https://assets.leetcode.com/uploads/2020/11/26/tmp-tree.jpg)

	Input: root = [3,9,20,null,null,15,7]
	Output: 3

	
Example 2:

	Input: root = [1,null,2]
	Output: 2


Example 3:

	Input: root = []
	Output: 0

	
Example 4:

	
	Input: root = [0]
	Output: 1


Note:

	
	The number of nodes in the tree is in the range [0, 10^4].
	-100 <= Node.val <= 100

### 解析


根据题意，就是给出了一棵二叉树的根节点 root ，要求我们返回这棵树的最大深度。题目中给出了二叉树的最大深度的定义，就是沿着从根节点到最远叶节点的最长路径的节点个数。其实碰到二叉树的常见问题，你下意识用递归的思路做基本上大概率是靠谱的。

思路比较简单，这里就是定义一个递归函数，如果该节点为空直接 0 ，如果不为空继续加一去探测其左右两边的子树深度，最后返回的结果就是该二叉树的最大深度。

### 解答
				
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def maxDepth(self, root):
	        """
	        :type root: TreeNode
	        :rtype: int
	        """
	        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1 if root else 0

            	      
			
### 运行结果

	Runtime: 36 ms, faster than 47.47% of Python online submissions for Maximum Depth of Binary Tree.
	Memory Usage: 16.2 MB, less than 32.54% of Python online submissions for Maximum Depth of Binary Tree.

### 解析

当然了也可以不用递归，直接遍历树的每一层节点，当到达新一层的时候最大深度加一，遍历完所有层，即可得到答案。

### 解答

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def maxDepth(self, root):
	        """
	        :type root: TreeNode
	        :rtype: int
	        """
	        result = 0
	        stack = [root] if root else []
	        while stack:
	            result += 1
	            tmp = []
	            for node in stack:
	                if node.left:
	                    tmp.append(node.left)
	                if node.right:
	                    tmp.append(node.right)
	            stack = tmp
	        return result

### 运行结果
	
	Runtime: 34 ms, faster than 49.72% of Python online submissions for Maximum Depth of Binary Tree.
	Memory Usage: 15.9 MB, less than 74.53% of Python online submissions for Maximum Depth of Binary Tree.

原题链接：https://leetcode.com/problems/maximum-depth-of-binary-tree/



您的支持是我最大的动力
