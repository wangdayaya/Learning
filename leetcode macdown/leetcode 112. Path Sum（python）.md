leetcode  112. Path Sum（python）

### 描述

Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

A leaf is a node with no children.



Example 1:

![](https://assets.leetcode.com/uploads/2021/01/18/pathsum1.jpg)

	Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
	Output: true

	
Example 2:


![](https://assets.leetcode.com/uploads/2021/01/18/pathsum2.jpg)

	Input: root = [1,2,3], targetSum = 5
	Output: false


Example 3:

	Input: root = [1,2], targetSum = 0
	Output: false

	



Note:


	The number of nodes in the tree is in the range [0, 5000].
	-1000 <= Node.val <= 1000
	-1000 <= targetSum <= 1000

### 解析

根据题意，就是给出了一个二叉树的根节点 root 和一个目标整数 targetSum ，如果这棵树有一条从根节点到叶子节点的路径的和等于 targetSum ，就返回 true ，否则返回 false 。

仍然使用递归解法，定义一个递归函数 dfs ，让其去深度遍历从 root 到某个 leaf 的路径存入列表 nodes 中，当到达叶子节点的时候比较列表 nodes 中的值的和是否等于 targetSum 。

### 解答
				

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def hasPathSum(self, root, targetSum):
	        """
	        :type root: TreeNode
	        :type targetSum: int
	        :rtype: bool
	        """
	        if not root:return False
	        def dfs(root, nodes):
	            if not root: return sum(nodes) == targetSum
	            if not root.left:
	                return dfs(root.right, nodes+[root.val])
	            if not root.right:
	                return dfs(root.left, nodes+[root.val])
	            if root.left and root.right:
	                return dfs(root.right, nodes+[root.val]) or dfs(root.left, nodes+[root.val])
	        return dfs(root, [])
            	      
			
### 运行结果


	Runtime: 28 ms, faster than 94.13% of Python online submissions for Path Sum.
	Memory Usage: 15.4 MB, less than 51.88% of Python online submissions for Path Sum.


### 解析

其实也可以不定义其他的递归函数，直接在 hasPathSum 函数上进行递归，原理和上面一样。

### 解答

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def hasPathSum(self, root, targetSum):
	        """
	        :type root: TreeNode
	        :type targetSum: int
	        :rtype: bool
	        """
	        if not root:return False
	        if not root.left and not root.right and root.val==targetSum:
	            return True
	        targetSum -= root.val
	        return self.hasPathSum(root.left, targetSum) or  self.hasPathSum(root.right, targetSum) 

### 运行结果

	Runtime: 65 ms, faster than 5.78% of Python online submissions for Path Sum.
	Memory Usage: 15.1 MB, less than 97.45% of Python online submissions for Path Sum.


原题链接：https://leetcode.com/problems/path-sum/



您的支持是我最大的动力
