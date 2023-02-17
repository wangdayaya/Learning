leetcode  783. Minimum Distance Between BST Nodes（python）




### 描述

Given the root of a Binary Search Tree (BST), return the minimum difference between the values of any two different nodes in the tree.





Example 1:

![](https://assets.leetcode.com/uploads/2021/02/05/bst1.jpg)

	Input: root = [4,2,6,1,3]
	Output: 1

	
Example 2:

![](https://assets.leetcode.com/uploads/2021/02/05/bst2.jpg)

	Input: root = [1,0,48,null,null,12,49]
	Output: 1


Note:


	The number of nodes in the tree is in the range [2, 100].
	0 <= Node.val <= 10^5

### 解析

根据题意，给定二叉搜索树的根 root ，返回树中任意两个不同节点的值之间的最小差值。这道题其实很简单，我们只需要遍历所有的节点的值保存到列表中，然后找出某两个节点值的最小差值就可以了。

N 为节点个数，时间复杂度为 O(N ，空间复杂度为 O(N) 。


### 解答

	class Solution(object):
	    def minDiffInBST(self, root):
	        self.vals = []
	        self.inOrder(root)
	        return min([self.vals[i + 1] - self.vals[i] for i in xrange(len(self.vals) - 1)])
	
	    def inOrder(self, root):
	        if not root:
	            return 
	        self.inOrder(root.left)
	        self.vals.append(root.val)
	        self.inOrder(root.right)
	 


### 运行结果

	Runtime 10 ms Beats 96.77%
	Memory 13.6 MB Beats 62.37%

### 原题链接
https://leetcode.com/problems/minimum-distance-between-bst-nodes/description/



您的支持是我最大的动力
