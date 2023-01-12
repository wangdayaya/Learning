leetcode  687. Longest Univalue Path（python）

### 每日经典

《如梦令》      ——李清照(宋) 

昨夜雨疏风骤，浓睡不消残酒。试问卷帘人，却道海棠依旧。知否，知否？应是绿肥红瘦。  


### 描述

Given the root of a binary tree, return the length of the longest path, where each node in the path has the same value. This path may or may not pass through the root.

The length of the path between two nodes is represented by the number of edges between them.


Example 1:


![](https://assets.leetcode.com/uploads/2020/10/13/ex1.jpg)

	Input: root = [5,4,5,1,1,5]
	Output: 2
	
Example 2:

![](https://assets.leetcode.com/uploads/2020/10/13/ex2.jpg)

	Input: root = [1,4,5,4,4,5]
	Output: 2



Note:


	The number of nodes in the tree is in the range [0, 10^4].
	-1000 <= Node.val <= 1000
	The depth of the tree will not exceed 1000.

### 解析

根据题意，给定二叉树的根节点 root ，返回最长路径的长度，其中路径中的每个节点都具有相同的值。 需要注意的是，此路径可能会也可能不会通过根，而且还有一个需要注意的是路径的长度是靠边数来决定的。假如有 n 个节点的路径，那么路径的长度为 n-1 。这个不重要，我们为了方便计算和理解，就将问题转化为求最长相同节点个数。

定义一个递归函数 DFS(root) ，返回以当前节点为根结点可以向左、右某个方向往下可以找到的最长相同节点的个数。调用递归函数，计算当前节点的左子树可能的最长相同节点个数 L ，和右子树可能的最长相同节点个数 R ，如果当前节点的值和孩子节点的值不同则将 L 或 R 重置为 0 ，使用 L+R+1 更新最长相同节点个数 result 即可，但是函数的返回值要返回左右任意一个较大的节点数 max(L,R)+1 。递归函数结束，我们得到的只是相同值路径的最大节点数，还需要减一，这样就可以表示边数。

### 解答
				

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def __init__(self):
	        self.result = 0
	    def longestUnivaluePath(self, root):
	        """
	        :type root: TreeNode
	        :rtype: int
	        """
	        if not root: return 0
	        self.dfs(root)
	        return self.result-1
	    
	    def dfs(self, root):
	        if not root: return 0
	        L = self.dfs(root.left)
	        R = self.dfs(root.right)
	        if root.left and root.left.val != root.val:
	            L = 0
	        if root.right and root.right.val != root.val:
	            R = 0
	        self.result = max(L+R+1, self.result)
	        return max(L,R)+1
            	      
			
### 运行结果

	
	Runtime: 432 ms, faster than 45.45% of Python online submissions for Longest Univalue Path.
	Memory Usage: 20.5 MB, less than 55.30% of Python online submissions for Longest Univalue Path.


原题链接：https://leetcode.com/problems/longest-univalue-path/



您的支持是我最大的动力
