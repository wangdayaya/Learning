leetcode  1740. Find Distance in a Binary Tree（python）

### 描述


Given the root of a binary tree and two integers p and q , return the distance between the nodes of value p and value q in the tree.

The distance between two nodes is the number of edges on path from one to the other.


Example 1:

![](https://assets.leetcode.com/uploads/2018/12/14/binarysearchtree_improved.png)

	Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 3, q = 5
	Output: 2


	

Note:

	The number of nodes in the tree is in the range [2, 10^5].
	-10^9 <= Node.val <= 10^9
	All Node.val are unique.
	p != q
	p and q will exist in the BST.


### 解析

根据题意，给定二叉树的 root 和两个整数 p 和 q ，返回树中值 p 和值 q 的节点之间的距离。两个节点之间的距离是从一个节点到另一个节点的路径上的边数。其实这道题考察的还是最低公共祖先 LCA 的问题，但是还比 LCA 还要再多一步计算过程，就是要算 LCA 到 p 的距离加上 LCA 到 q 的距离。

如果用单纯暴力的算法，也是可以解决的，还是老样子用 DFS 找出从根节点到 p 的路径和根结点到 q 的路径，然后同时从遍历两个路径找到开始分叉的节点，那就是 LCA ，然后再算出 LCA 到 p 以及 LCA 到 q 的距离相加即可。

但是这道题还可以从另外一个角度直接使用 DFS 进行解题，这样的话可以更加节省空间，我们定义 DFS(root, p, q) 函数，返回的是两个整数，第一个返回值是 root 到 p 的距离，第二个返回值是 root 到 q 的距离。如果 root 下面没有 p 或者 q ，那么对应的结果为 -1 。从根节点不断递归左右两个子树，因为返回结果是从下往上的，所以在第一次出现以某个节点为 root 的子树根节点到 p 和 q 的距离都不为 -1 ，那么该节点为 LCA ，将这两个的距离相加就是答案。具体细节看代码。

### 解答
		
	class TreeNode(object):
	    def __init__(self, x):
	        self.val = x
	        self.left = None
	        self.right = Nones
		
	class Solution(object):
	    def __init__(self):
	        self.result = None
	    def lowestCommonAncestor(self, root, p, q):
	        self.dfs(root, p, q)
	        return self.result
	
	    def dfs(self, root, p, q):
	        if not root: return [-1,-1]
	        if self.result: return [-1,-1]
	        L = self.dfs(root.left, p, q)
	        R = self.dfs(root.right, p, q)
	
	        dis_L = -1
	        dis_R = -1
	        if L[0] != -1:
	            dis_L = L[0]+1
	        elif R[0] != -1:
	            dis_L = R[0]+1
	        elif root.val == p:
	            dis_L = 0
	
	        if L[1] != -1:
	            dis_R = L[1]+1
	        elif R[1] != -1:
	            dis_R = R[1]+1
	        elif root.val == q:
	            dis_R = 0
	
	        if dis_L!=-1 and dis_R!=-1 and not self.result:
	            self.result = dis_L + dis_R
	
	        return [dis_L, dis_R]

            	      
			
### 运行结果

	Runtime: 72 ms, faster than 55.76% of Python online submissions for Find Distance in a Binary Tree.
	Memory Usage: 21.4 MB, less than 55.96% of Python online submissions for Find Distance in a Binary Tree.
	
	

原题链接：https://leetcode.com/problems/find-distance-in-a-binary-tree/



您的支持是我最大的动力
