leetcode 105. Construct Binary Tree from Preorder and Inorder Traversal （python）




### 描述


Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.




Example 1:

![](https://assets.leetcode.com/uploads/2021/02/19/tree.jpg)

	Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
	Output: [3,9,20,null,null,15,7]

	
Example 2:



	Input: preorder = [-1], inorder = [-1]
	Output: [-1]
Example 3:





Note:

	1 <= preorder.length <= 3000
	inorder.length == preorder.length
	-3000 <= preorder[i], inorder[i] <= 3000
	preorder and inorder consist of unique values.
	Each value of inorder also appears in preorder.
	preorder is guaranteed to be the preorder traversal of the tree.
	inorder is guaranteed to be the inorder traversal of the tree.


### 解析

根据题意，给定两个整数数组 preorder 和 inorder，其中 preorder 是二叉树的前序遍历结果，inorder 是同一棵树的中序遍历结果，构造并返回二叉树。

首先我们要知道前序遍历的定义和中序遍历的定义，然后我们找规律，对于一棵新的树，其前序遍历总是：

	[ 根, [左子树的前序遍历], [右子树的前序遍历] ]

其中序遍历总是：

	[ [左子树的中序遍历], 根, [右子树的中序遍历] ]

所以我们可以使用递归的方式进行解题，前序遍历总左往右的每个值都是一个新子树的根，每次只要从前序遍历中找出根，然后再去中序遍历中找其左右子树即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答

	class Solution:
	    def buildTree(self, P: List[int], I: List[int]) -> Optional[TreeNode]:
	        d = {c: i for i, c in enumerate(I)}
	        index = 0
	        def findRoot(L, R):
	            nonlocal index
	            if L > R:
	                return None
	            v = P[index]
	            index += 1
	            root = TreeNode(v)
	            root.left = findRoot(L, d[v] - 1)
	            root.right = findRoot(d[v] + 1, R)
	            return root
	
	        return findRoot(0, len(P) - 1)
### 运行结果

	Runtime: 124 ms, faster than 72.80% of Python3 online submissions for Construct Binary Tree from Preorder and Inorder Traversal.
	Memory Usage: 18.9 MB, less than 76.63% of Python3 online submissions for Construct Binary Tree from Preorder and Inorder Traversal.



### 原题链接

https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/


您的支持是我最大的动力
