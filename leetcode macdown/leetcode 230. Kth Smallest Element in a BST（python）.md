leetcode  230. Kth Smallest Element in a BST（python）




### 描述

Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.



Example 1:


![](https://assets.leetcode.com/uploads/2021/01/28/kthtree1.jpg)

	Input: root = [3,1,4,null,2], k = 1
	Output: 1	
Example 2:


![](https://assets.leetcode.com/uploads/2021/01/28/kthtree2.jpg)

	Input: root = [5,3,6,2,4,null,null,1], k = 3
	Output: 3




Note:


	The number of nodes in the tree is n.
	1 <= k <= n <= 10^4
	0 <= Node.val <= 10^4

### 解析


根据题意，给定二叉搜索树的根和一个整数 k ，返回树中所有节点值的第 k 个最小值（索引从 1 开始）。

这道题虽然难度是 Medium ，但是其实看过题目我们就能瞬间有了解题思路，就是考察对树的中序遍历，中序遍历其实就是先遍历左节点，再遍历根节点，再遍历右节点的顺序访问二叉树。

既然我们知道了题目考查的内容是中序遍历，那么最常见的就是两种方法，一种是递归解法，另一种就是用栈来实现，这里的解法是第一种解法。

递归的思路就是中序遍历的定义很简单，按照中序遍历的顺序将所有节点的值都放入一个列表中，最后用 k-1 索引返回值即可。

时间复杂度为 O(N)，空间复杂度为 O(N)  。

另外题目还给出了更加复杂的条件让有能力的同学做，如果 BST 经常被修改（即我们可以进行插入和删除节点操作），并且需要让你频繁地找到第 k 个最小的值，你会如何优化？有兴趣的同学可以思考一下。

### 解答
				
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def kthSmallest(self, root, k):
	        """
	        :type root: TreeNode
	        :type k: int
	        :rtype: int
	        """
	        def inorder(root):
	            result = []
	            if not root: return result
	            for node in inorder(root.left):
	                result.append(node)
	            result.append(root.val)
	            for node in inorder(root.right):
	                result.append(node)
	            return result
	        return inorder(root)[k-1]

            	      
			
### 运行结果

	Runtime: 63 ms, faster than 36.13% of Python online submissions for Kth Smallest Element in a BST.
	Memory Usage: 21.3 MB, less than 29.74% of Python online submissions for Kth Smallest Element in a BST.


### 原题链接


https://leetcode.com/problems/kth-smallest-element-in-a-bst/


您的支持是我最大的动力
