leetcode 98. Validate Binary Search Tree （python）




### 描述

Given the root of a binary tree, determine if it is a valid binary search tree (BST). A valid BST is defined as follows:

* The left subtree of a node contains only nodes with keys less than the node's key.
* The right subtree of a node contains only nodes with keys greater than the node's key.
* Both the left and right subtrees must also be binary search trees.




Example 1:

![](https://assets.leetcode.com/uploads/2020/12/01/tree1.jpg)

	Input: root = [2,1,3]
	Output: true

	
Example 2:


![](https://assets.leetcode.com/uploads/2020/12/01/tree2.jpg)

	Input: root = [5,1,4,null,null,3,6]
	Output: false
	Explanation: The root node's value is 5 but its right child's value is 4.

Example 3:





Note:


	The number of nodes in the tree is in the range [1, 10^4].
	-2^31 <= Node.val <= 2^31 - 1

### 解析

根据题意，给定二叉树的根 root ，确定它是否是有效的二叉搜索树 (BST)。有效的 BST 定义如下：

* 节点的左子树仅包含键小于节点键的节点。
* 节点的右子树仅包含键大于节点键的节点。
* 左右子树也必须是二叉搜索树。

最简单的办法就是使用一个列表 L ，然后使用递归的方式进行中序遍历，这样我们就能得到一个中序遍历的节点值，然后判断这个 L 中的值如果是递增的顺序那么说明是有效的 BST ，否则就是无效的 BST 。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def isValidBST(self, root):
	        """
	        :type root: TreeNode
	        :rtype: bool
	        """
	        L = []
	        def dfs(root):
	            if not root:
	                return
	            dfs(root.left)
	            L.append(root.val)
	            dfs(root.right)
	
	        dfs(root)
	        for i in range(1, len(L)):
	            if L[i] <= L[i-1]:
	                return False
	        return True

### 运行结果

	Runtime: 57 ms, faster than 35.37% of Python online submissions for Validate Binary Search Tree.
	Memory Usage: 18.4 MB, less than 16.26% of Python online submissions for Validate Binary Search Tree.


### 解析

另外还可以使用栈来保存中序遍历的节点，而且我们可以使用变量 pre 来表示当前节点的左子树的最大值，在进行中序遍历的过程中，只要 pre 大于或者等于当前节点值就说明是不合法的 BST 直接返回 False 即可。如果遍历正常结束说明是合法的 BST 直接返回  True 。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。
### 解答

	class Solution(object):
	    def isValidBST(self, root):
	        """
	        :type root: TreeNode
	        :rtype: bool
	        """
	        stack = []
	        pre = -float('inf')
	        while stack or root:
	            while root:
	                stack.append(root)
	                root = root.left
	            root = stack.pop()
	            if pre >= root.val:
	                return False
	            pre = root.val
	            root = root.right
	        return True

### 运行结果

	Runtime: 24 ms, faster than 98.73% of Python online submissions for Validate Binary Search Tree.
	Memory Usage: 18.1 MB, less than 45.60% of Python online submissions for Validate Binary Search Tree.
### 原题链接

https://leetcode.com/problems/validate-binary-search-tree/


您的支持是我最大的动力
