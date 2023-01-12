leetcode  236. Lowest Common Ancestor of a Binary Tree（python）

### 描述

Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”


Example 1:

![](https://assets.leetcode.com/uploads/2018/12/14/binarytree.png)

	Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
	Output: 3
	Explanation: The LCA of nodes 5 and 1 is 3.

Example 2:

![](https://assets.leetcode.com/uploads/2018/12/14/binarytree.png)

	Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
	Output: 5
	Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.

Example 3:

	Input: root = [1,2], p = 1, q = 2
	Output: 1


Note:

	The number of nodes in the tree is in the range [2, 10^5].
	-10^9 <= Node.val <= 10^9
	All Node.val are unique.
	p != q
	p and q will exist in the tree.


### 解析

根据题意，给定一棵二叉树，找出树中两个给定节点的最低共同祖先（LCA）。根据维基百科 LCA 的定义：最低共同祖先被定义为两个节点 p 和 q 之间的最低节点，即 T 中同时具有 p 和 q 作为后代的节点（我们允许一个节点是它自己的后代）。看了几个例子之后应该就会理解 LCA 的含义了。这道题最简单的解法就是暴力找到根节点到 p 的路径和根节点到 q 的路径，然后找出分叉的节点，那就是 LCA ，但是这种解法时间复杂度在于找 p 或者 q ，时间复杂度为 O(n) ，尽管可以成功通过，但是因为要存两条途径，空间复杂度会变成  O(logn)。

### 解答
				

	class TreeNode(object):
	    def __init__(self, x):
	        self.val = x
	        self.left = None
	        self.right = None
	
	class Solution(object):
	    def lowestCommonAncestor(self, root, p, q):
	        """
	        :type root: TreeNode
	        :type p: TreeNode
	        :type q: TreeNode
	        :rtype: TreeNode
	        """
	        result = []
	        def dfs(root, path, target):
	            if not root:
	                return 
	            if root.val == target.val:
	                path.append(root.val)
	                result.append(path)
	            if root.left:
	                dfs(root.left, path+[root.val], target)
	            if root.right:
	                dfs(root.right, path+[root.val], target)
	        dfs(root, [], p)
	        dfs(root, [], q)
	        p_path = result[0]
	        q_path = result[1]
	        idx = 0
	        while idx < min(len(p_path), len(q_path)):
	            if p_path[idx] != q_path[idx]:
	                return TreeNode(p_path[idx-1])
	            idx += 1
	        return TreeNode(p_path[idx-1])
            	      
			
### 运行结果

	Runtime: 2456 ms, faster than 5.01% of Python online submissions for Lowest Common Ancestor of a Binary Tree.
	Memory Usage: 418.2 MB, less than 5.06% of Python online submissions for Lowest Common Ancestor of a Binary Tree.


### 解析

另外我们可以定义 dfs 为某个子树中包含了多少个 p 或者 q ，如果没有包含 p 或者 q 则返回 0 ，如果只包含了 p 或者 q 中的任意一个返回 1 ，如果都包含则返回 2 。因为是顶部向下递归，返回结果的时候是从下往上，所以当我们第一次碰到 dfs 返回 2 的某个节点，则肯定是 LCA 。


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

	Runtime: 68 ms, faster than 63.65% of Python online submissions for Lowest Common Ancestor of a Binary Tree.
	Memory Usage: 29.9 MB, less than 8.97% of Python online submissions for Lowest Common Ancestor of a Binary Tree.
	
原题链接：https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/



您的支持是我最大的动力
