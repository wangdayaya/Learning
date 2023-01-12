leetcode  1123. Lowest Common Ancestor of Deepest Leaves（python）

### 描述


Given the root of a binary tree, return the lowest common ancestor of its deepest leaves.

Recall that:

* The node of a binary tree is a leaf if and only if it has no children
* The depth of the root of the tree is 0. if the depth of a node is d, the depth of each of its children is d + 1.
* The lowest common ancestor of a set S of nodes, is the node A with the largest depth such that every node in S is in the subtree with root A.


Note: This question is the same as 865: https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes/




Example 1:


![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/01/sketch1.png)

	Input: root = [3,5,1,6,2,0,8,null,null,7,4]
	Output: [2,7,4]
	Explanation: We return the node with value 2, colored in yellow in the diagram.
	The nodes coloured in blue are the deepest leaf-nodes of the tree.
	Note that nodes 6, 0, and 8 are also leaf nodes, but the depth of them is 2, but the depth of nodes 7 and 4 is 3.
	
Example 2:


	Input: root = [1]
	Output: [1]
	Explanation: The root is the deepest node in the tree, and it's the lca of itself.

Example 3:

	Input: root = [0,1,3,null,2]
	Output: [2]
	Explanation: The deepest leaf node in the tree is 2, the lca of one node is itself.

	
Note:

	The number of nodes in the tree will be in the range [1, 1000].
	0 <= Node.val <= 1000
	The values of the nodes in the tree are unique.


### 解析


根据题意，就是给出了一个二叉树的根节点引用 root ，让我们找出最低的共同祖先节点（lowest common ancestor）。结合题目描述和多个例子我们可以总结出来，有三种结果：

* 如果只有根结点，那它本身就是最低的共同祖先节点，如例二；
* 如果某两个最深的兄弟结点的深度相同，那么两者的共同父节点为最低的共同祖先节点，如例一；
* 如果只有一个叶子结点的深度是全树最深的，那么该结点为最低的共同祖先节点，如例三；

所以使用递归思想，定义递归函数 dfs ，每次调用 dfs 返回当前深度的最低的共同祖先节点引用以及当前的深度。


### 解答
				

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def lcaDeepestLeaves(self, root):
	        """
	        :type root: TreeNode
	        :rtype: TreeNode
	        """
	        def dfs(root):
	            if not root:
	                return None, 0
	
	            L, L_d = dfs(root.left)
	            R, R_d = dfs(root.right)
	
	            if L_d == R_d:
	                return root, L_d+1 
	            elif L_d > R_d:
	                return L, L_d+1
	            else:
	                return R, R_d+1
	
	        return dfs(root)[0]
            	      
			
### 运行结果

	Runtime: 40 ms, faster than 66.99% of Python online submissions for Lowest Common Ancestor of Deepest Leaves.
	Memory Usage: 14.1 MB, less than 7.77% of Python online submissions for Lowest Common Ancestor of Deepest Leaves.

### 解析

换一种不同的写法，原理一样

### 解答


	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def lcaDeepestLeaves(self, root):
	        """
	        :type root: TreeNode
	        :rtype: TreeNode
	        """
	        def dfs(root, d):
	            if not root: return None,d
	
	            L, L_d = dfs(root.left, d+1)        
	            R, R_d = dfs(root.right, d+1)
	
	            if L_d == R_d:
	                return root, L_d
	            elif L_d > R_d:
	                return L, L_d
	            else:
	                return R, R_d
	        return dfs(root, 0)[0]

### 运行结果

	Runtime: 36 ms, faster than 91.26% of Python online submissions for Lowest Common Ancestor of Deepest Leaves.
	Memory Usage: 13.9 MB, less than 27.18% of Python online submissions for Lowest Common Ancestor of Deepest Leaves.
	
	
### 解析

另外可以换一种比较朴素的思路，先使用 DFS 遍历一次二叉树，找出树中叶子节点的最大深度以及最大深度的所有叶子结点的个数，然后再用另一个 DFS 遍历二叉树，如果第一次出现以某个节点为根节点的子树包含了所有的最深叶子节点，那么它就是答案。

### 解答

	# Definition for a binary tree node.
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	        
	class Solution(object):
	    def __init__(self):
	        self.maxDepth = 0
	        self.maxDepthCount = 0
	        self.result = None
	        self.valToNum = {}
	        
	    def lcaDeepestLeaves(self, root):
	        """
	        :type root: TreeNode
	        :rtype: TreeNode
	        """
	        self.dfs1(root, 0)
	        self.maxDepth = max(self.valToNum.values())
	        for k,v in self.valToNum.items():
	            self.maxDepthCount += 1 if v==self.maxDepth else 0
	        self.dfs2(root)
	        return self.result
	    
	    def dfs1(self, root, depth):
	        if not root: return 
	        self.valToNum[root.val] = depth
	        self.dfs1(root.left, depth + 1)
	        self.dfs1(root.right, depth + 1)
	        
	    def dfs2(self, root):
	        if not root: return 0
	        self_ = 1 if self.valToNum[root.val]==self.maxDepth else 0
	        a = self.dfs2(root.left)
	        b = self.dfs2(root.right)
	        if self_ + a + b == self.maxDepthCount and not self.result:
	            self.result = root
	        return self_ + a + b
	        

### 运行结果

	Runtime: 52 ms, faster than 16.54% of Python online submissions for Lowest Common Ancestor of Deepest Leaves.
	Memory Usage: 14.3 MB, less than 12.03% of Python online submissions for Lowest Common Ancestor of Deepest Leaves.

原题链接：https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/



您的支持是我最大的动力
