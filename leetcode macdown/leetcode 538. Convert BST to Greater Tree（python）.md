leetcode  538. Convert BST to Greater Tree（python）




### 描述



Given the root of a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus the sum of all keys greater than the original key in BST.

As a reminder, a binary search tree is a tree that satisfies these constraints:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.

Example 1:

![](https://assets.leetcode.com/uploads/2019/05/02/tree.png)
	
	Input: root = [4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
	Output: [30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]

	
Example 2:


	Input: root = [0,null,1]
	Output: [1,null,1]






Note:

	The number of nodes in the tree is in the range [0, 10^4].
	-10^4 <= Node.val <= 10^4
	All the values in the tree are unique.
	root is guaranteed to be a valid binary search tree.


### 解析

这道题其实和 [leetcode 1038. Binary Search Tree to Greater Sum Tree](https://leetcode.com/problems/binary-search-tree-to-greater-sum-tree/) 是一样的。


根据题意，给定二叉搜索树的根 root ，将其转换为更大的树，使得原始 BST 的每个节点的值都更改为原始值加上所有大于其原始键的节点值的总和。

题意是相当简单，如果觉得不清楚可以结合例子理解，很容易明白的。

这道题就是考察使用 DFS 对树的节点进行逆序遍历，因为二叉搜索树特有的特点，也就是右子树的值肯定是大于根节点的，左子树的值肯定是小于根节点的，我们可以换个角度看这个问题，因为上述二叉搜索树特有的性质，将树的左端到树的右端所有元素拉成一条直线，我们发现其实就是一个升序的数组，我们的目标就是逆序遍历这个数组并且求累加和，并更新当前位置上的节点值，这么一来思路就很清晰了：

* 定义一个全局的变量 total 用来进行累加和
* 想在树中实现遍历所有节点的顺序是逆向的过程只能使用 DFS ，所以递归进行 DFS 遍历所有节点，在更新 total 的同时也更新当前节点的值
* 结束之后返回根节点 root 即可

时间复杂度为 O(N) ，因为调用了 N 次 convertBST ，空间复杂度为 O(N)，因为最坏的情况可能有 N 个右（或左）节点，递归调用的栈可能会有 N 层。



### 解答
				

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def __init__(self):
	        self.total = 0
	    def convertBST(self, root):
	        """
	        :type root: TreeNode
	        :rtype: TreeNode
	        """
	        if not root:return 
	        self.convertBST(root.right)
	        self.total += root.val
	        root.val = self.total
	        self.convertBST(root.left)
	        return root
            	      
			
### 运行结果

	Runtime: 56 ms, faster than 99.19% of Python online submissions for Convert BST to Greater Tree.
	Memory Usage: 17.8 MB, less than 52.85% of Python online submissions for Convert BST to Greater Tree.


### 原题链接


https://leetcode.com/problems/convert-bst-to-greater-tree/


您的支持是我最大的动力
