leetcode  110. Balanced Binary Tree（python）


### 描述


Given a binary tree, determine if it is height-balanced.

For this problem, a height-balanced binary tree is defined as:

a binary tree in which the left and right subtrees of every node differ in height by no more than 1.


Example 1:

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/64d4a623bb7042268e49ada77a52f8db~tplv-k3u1fbpfcp-zoom-1.image)

	Input: root = [3,9,20,null,null,15,7]
	Output: true

	
Example 2:


![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5dcaba7d0c414aff804788f96014bb28~tplv-k3u1fbpfcp-zoom-1.image)

	Input: root = [1,2,2,3,3,null,null,4,4]
	Output: false

Example 3:


	Input: root = []
	Output: true
	

Note:


	The number of nodes in the tree is in the range [0, 5000].
	-104 <= Node.val <= 104

### 解析


根据题意，给出了一个二叉树，检查这个二叉树是不是高度平衡的。

高度平衡的二叉树，就是指每个节点的左子树和右子树的高度相差不超过 1。

这道题其实和 [leetcode 108. Convert Sorted Array to Binary Search Tree](https://juejin.cn/post/7021322404328636452) 考察的内容一样，都是计算树的深度，那道题都会做这道题也是类似的。

定义一个函数 depth 用来计算某个节点的高度，如果节点为空就直接返回 0 。如果不为空则继续深入它的左右两个子树计算当前节点的高度。

定义一个函数 isBalanced 用来判断当前节点是否是平衡的二叉树，通过 depth 可以得到其左右子树的高度 l 和 r ，然后比较当前节点是否左右子树的高度差小于 2 且左右两个子树也是高度平衡二叉树。



### 解答
				
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def isBalanced(self, root):
	        """
	        :type root: TreeNode
	        :rtype: bool
	        """
	        if not root:return True
	        def depth(root):
	            if not root:return 0
	            return max(depth(root.left), depth(root.right)) + 1
	        l = depth(root.left)
	        r = depth(root.right)
	        return abs(l-r)<=1 and self.isBalanced(root.left) and self.isBalanced(root.right)
	
	            
	        
			
### 运行结果

	Runtime: 56 ms, faster than 48.18% of Python online submissions for Balanced Binary Tree.
	Memory Usage: 17.6 MB, less than 88.07% of Python online submissions for Balanced Binary Tree.


### 类似题目

* [leetcode 108. Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)

原题链接：https://leetcode.com/problems/balanced-binary-tree/


您的支持是我最大的动力 !!!
