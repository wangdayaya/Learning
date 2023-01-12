leetcode  101. Symmetric Tree（python）

### 描述


Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).


Example 1:

![](https://assets.leetcode.com/uploads/2021/02/19/symtree1.jpg)

	Input: root = [1,2,2,3,4,4,3]
	Output: true


	
Example 2:


![](https://assets.leetcode.com/uploads/2021/02/19/symtree2.jpg)

	Input: root = [1,2,2,null,3,null,3]
	Output: false





Note:

	The number of nodes in the tree is in the range [1, 1000].
	-100 <= Node.val <= 100


### 解析

根据题意，给出了一个二叉树，判断这个树是不是根据中线对称的。一开始思路比较狭窄，光想着对树本身是用递归，发现很难，最后看到了大神的解法，清晰且容易。

思路比较简单，就是定义一个递归函数 symmetric ，有两个参数 left 和 right ，分别代表了两个树的根节点
在 symmetric 函数中传入两个相同的 root ，对两个相同的树进行操作，判断 left 的值是否和 right 的值是否相等，如果相等继续递归 symmetric(left.left, right.right) and symmetric(left.right, right.left) ，如果不相等直接返回 False 即可。

### 解答
				

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def isSymmetric(self, root):
	        """
	        :type root: TreeNode
	        :rtype: bool
	        """
	        def symmetric(left, right):
	            if not left and not right: return True
	            if left and right and left.val == right.val:
	                return symmetric(left.left, right.right) and symmetric(left.right, right.left)
	            return False
	        return symmetric(root, root)
            	      
			
### 运行结果
	
	Runtime: 20 ms, faster than 84.75% of Python online submissions for Symmetric Tree.
	Memory Usage: 13.6 MB, less than 43.91% of Python online submissions for Symmetric Tree.


### 解析

当然同样可以使用栈来迭代树中的节点来进行比较。

### 解答

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def isSymmetric(self, root):
	        """
	        :type root: TreeNode
	        :rtype: bool
	        """
	        if not root:
	            return True
	        stack = [(root.left, root.right)]
	        while stack:
	            l, r = stack.pop()
	            if not l and not r:
	                continue
	            if not l or not r or (l.val != r.val):
	                return False
	            stack.append((l.left, r.right))
	            stack.append((l.right, r.left))
	        return True




### 运行结果

	Runtime: 40 ms, faster than 12.21% of Python online submissions for Symmetric Tree.
	Memory Usage: 13.6 MB, less than 70.59% of Python online submissions for Symmetric Tree.
原题链接：https://leetcode.com/problems/symmetric-tree/



您的支持是我最大的动力
