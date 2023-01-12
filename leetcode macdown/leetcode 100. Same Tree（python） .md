leetcode  100. Same Tree（python）

### 描述

Given the roots of two binary trees p and q, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.





Example 1:

![](https://assets.leetcode.com/uploads/2020/12/20/ex1.jpg)
	
	Input: p = [1,2,3], q = [1,2,3]
	Output: true

	
Example 2:

![](https://assets.leetcode.com/uploads/2020/12/20/ex2.jpg)
	
	Input: p = [1,2], q = [1,null,2]
	Output: false



Example 3:

![](https://assets.leetcode.com/uploads/2020/12/20/ex3.jpg)

	Input: p = [1,2,1], q = [1,1,2]
	Output: false

	



Note:

	The number of nodes in both trees is in the range [0, 100].
	-10^4 <= Node.val <= 10^4


### 解析


根据题意，给出了两颗树 p 和 q ，要求判断这两个树是否完全一样。直接使用递归对两颗树的节点进行遍历，如果有一个节点不相同直接返回 False ，否则就返回 True 。

### 解答
				

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def isSameTree(self, p, q):
	        """
	        :type p: TreeNode
	        :type q: TreeNode
	        :rtype: bool
	        """
	        if not p and not q: return True
	        if not p or not q: return False
	        if p.val != q.val: return False
	        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
	        
            	      
			
### 运行结果

	Runtime: 16 ms, faster than 82.88% of Python online submissions for Same Tree.
	Memory Usage: 13.5 MB, less than 67.93% of Python online submissions for Same Tree.


### 解析

当然了还可以使用栈的结构来遍历所有的节点，逻辑和上面的类似。


### 解答


	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def isSameTree(self, p, q):
	        """
	        :type p: TreeNode
	        :type q: TreeNode
	        :rtype: bool
	        """
	        def isEqual(p,q):
	            if not p and not q: return True
	            if not p or not q: return False
	            if p.val != q.val: return False
	            return True
	        stack = [[p,q]]
	        while stack:
	            p,q = stack.pop(0)
	            if not isEqual(p,q):
	                return False
	            if p and q:
	                stack.append([p.left, q.left])
	                stack.append([p.right, q.right])
	        return True


### 运行结果

	Runtime: 16 ms, faster than 82.88% of Python online submissions for Same Tree.
	Memory Usage: 13.4 MB, less than 89.64% of Python online submissions for Same Tree.

原题链接：https://leetcode.com/problems/same-tree/



您的支持是我最大的动力
