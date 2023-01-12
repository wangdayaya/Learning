leetcode  235. Lowest Common Ancestor of a Binary Search Tree（python）

### 描述

Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

 



Example 1:


![](https://assets.leetcode.com/uploads/2018/12/14/binarysearchtree_improved.png)

	Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
	Output: 6
	Explanation: The LCA of nodes 2 and 8 is 6.
	
Example 2:

![](https://assets.leetcode.com/uploads/2018/12/14/binarysearchtree_improved.png)

	Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
	Output: 2
	Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.


Example 3:

	Input: root = [2,1], p = 2, q = 1
	Output: 2

	



Note:

	The number of nodes in the tree is in the range [2, 10^5].
	-10^9 <= Node.val <= 10^9
	All Node.val are unique.
	p != q
	p and q will exist in the BST.


### 解析

根据题意，给定一个二叉搜索树 (BST)，找到 BST 中两个给定节点的最低共同祖先 (LCA)。可以用最朴素的方法，找出从根节点到 p 的路径，也找出从根节点到 q 的路径，从左往右同时遍历两个路径的节点，找到第一次出现不同节点的前一个节点就是题目要找的 LCA 。因为题目中的限制条件很宽松，所以这种方法不会超时。


### 解答
				
	class TreeNode(object):
	    def __init__(self, x):
	        self.val = x
	        self.left = None
	        self.right = None
	
	class Solution(object):
	    def __init__(self):
	        self.paths = []
	    def lowestCommonAncestor(self, root, p, q):
	        """
	        :type root: TreeNode
	        :type p: TreeNode
	        :type q: TreeNode
	        :rtype: TreeNode
	        """
	        
	        self.dfs(root, [], p)
	        self.dfs(root, [], q)
	        path_p = self.paths[0]
	        path_q = self.paths[1]
	        idx = 0
	        while idx<min(len(path_p), len(path_q)) and path_p[idx].val==path_q[idx].val:
	            idx += 1
	        return path_p[idx-1]
	    
	    def dfs(self, root, path, t):
	        if not root: return
	        if root == t:
	            path.append(root)
	            self.paths.append(path)
	            return 
	        if root.left:
	            self.dfs(root.left, path+[root], t)
	        if root.right:
	            self.dfs(root.right, path+[root], t)
	            
	
	        
	
	           
            	      
			
### 运行结果

	Runtime: 96 ms, faster than 13.02% of Python online submissions for Lowest Common Ancestor of a Binary Search Tree.
	Memory Usage: 21.6 MB, less than 26.58% of Python online submissions for Lowest Common Ancestor of a Binary Search Tree.


### 解析

另外我们可以定义一个 dfs ，表示以某个节点为根节点的子树包含的 p 或者 q 的个数，可能为 0 表示没有包含 p 或者 q ，可能为 1 表示只包含了 p 或者 q ，可能为 2 表示都包含，因为递归是从下往上返回结果，当第一次出现 2 的时候并且 result 为空的时候，该节点为 LCA 。


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
	        self.dfs(root, p, q)
	        return self.result
	    
	    def dfs(self, root, p, q):
	        if not root: return 0
	        count  = self.dfs(root.left, p, q) + self.dfs(root.right, p, q) 
	        if root == p or root == q: 
	            count += 1
	        if count == 2 and self.result==None:
	            self.result = root
	        return count
	            
	
	        
### 运行结果

	
	Runtime: 72 ms, faster than 55.76% of Python online submissions for Lowest Common Ancestor of a Binary Search Tree.
	Memory Usage: 21.4 MB, less than 55.96% of Python online submissions for Lowest Common Ancestor of a Binary Search Tree.

原题链接：https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/



您的支持是我最大的动力
