

### 描述

Given the root of a binary tree, return the  lowest common ancestor (LCA) of two given nodes p and q. if either node p or q does not exist in the tree, return null. All values of the nodes in the tree are unique.

According to the definition of LCA on Wikipedia: The lowest common ancestor of two nodes p and q  in a binary tree T is the  lowest node that has both p and q as descendants(where we allow a node to be a descendant of itself), A desendant of a node x is a node y that is on the path from node x to some leaf node.




Example 1:


![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/20c97032be7e4323aced6829a2aad5f5~tplv-k3u1fbpfcp-zoom-1.image)

	Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
	Output: 3
	Explanation: The LCA of nodes 5 and 1 is 3.
	





Note

	The number of nodes in the tree is in the range [2, 10^5].
	-10^9 <= Node.val <= 10^9
	All Node.val are unique.
	p != q
	p and q will exist in the tree.



### 解析

根据题意，给定二叉树的根，返回两个给定节点 p 和 q 的最低共同祖先（LCA）。 如果树中不存在节点 p 或 q ，则返回 null 。题目会保证树中节点的所有值都是唯一的。

这道题其实和 【236. Lowest Common Ancestor of a Binary Tree 】的思路基本上是一样的，如果那个会做，这道题也肯定会做，有不懂的同学可以翻过去看一下。


### 解答
				
	
	class TreeNode(object):
	    def __init__(self, x):
	        self.val = x
	        self.left = None
	        self.right = None
	
	class Solution(object):
	    def __init__(self):
	        self.result = None
	    def lowestCommonAncestor(self, root, p, q):
	        """
	        :type root: TreeNode
	        :type p: TreeNode
	        :type q: TreeNode
	        :rtype: TreeNode
	        """
	        def dfs(root, p, q):
	            if not root: return 0
	            left = dfs(root.left, p, q)
	            right = dfs(root.right, p, q)
	            self_ = 1 if (p == root or q == root) else 0
	            count = left + right + self_
	            if count == 2 and not self.result:
	                self.result = root
	            return count
	        dfs(root, p, q)
	        return self.result
            	      
			
### 运行结果

	Runtime: 68 ms, faster than 63.65% of Python online submissions for Lowest Common Ancestor of a Binary Tree II.
	Memory Usage: 29.9 MB, less than 8.97% of Python online submissions for Lowest Common Ancestor of a Binary Tree II . 


原题链接：https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-ii/



您的支持是我最大的动力
