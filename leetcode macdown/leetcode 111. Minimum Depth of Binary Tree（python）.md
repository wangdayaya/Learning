leetcode  111. Minimum Depth of Binary Tree（python）

### 描述


Given a binary tree, find its minimum depth.

The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

Note: A leaf is a node with no children.




Example 1:

![](https://assets.leetcode.com/uploads/2020/10/12/ex_depth.jpg)

	Input: root = [3,9,20,null,null,15,7]
	Output: 2

	
Example 2:

	Input: root = [2,null,3,null,4,null,5,null,6]
	Output: 5



Note:

	The number of nodes in the tree is in the range [0, 10^5].
	-1000 <= Node.val <= 1000


### 解析

根据题意，就是给出了一棵二叉树，要找到它的最小的深度，题目中给出了最小深度的定义，就是从根节点到最近的叶节点沿最短路径的节点个数。还是按照常规套路使用递归的方法进行求解。

思路比较简单：

* 如果 root 为空则直接返回 0 
* 如果 root.left 为空则直接对 root.right 子树进行最小深度的查找
* 同理，如果root.right 为空则直接对 root.left 子树进行最小深度的查找
* 如果左右子树都不为空则对两个子树的深度求较小值即可。


### 解答
				
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def minDepth(self, root):
	        """
	        :type root: TreeNode
	        :rtype: int
	        """
	        if not root: return 0
	        if not root.left:
	            return self.minDepth(root.right) + 1
	        if not root.right:
	            return self.minDepth(root.left) + 1
	        l = self.minDepth(root.left) + 1
	        r = self.minDepth(root.right) + 1
	        return min(l, r)
	        

            	      
			
### 运行结果

	Runtime: 1167 ms, faster than 19.36% of Python online submissions for Minimum Depth of Binary Tree.
	Memory Usage: 95 MB, less than 22.36% of Python online submissions for Minimum Depth of Binary Tree.

### 解析

还可以使用栈的数据结构对节点进行迭代，找出最小的深度。
### 解答
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def minDepth(self, root):
	        """
	        :type root: TreeNode
	        :rtype: int
	        """
	        if not root:
	            return 0
	        queue = collections.deque([(root, 1)])
	        while queue:
	            node, level = queue.popleft()
	            if node:
	                if not node.left and not node.right:
	                    return level
	                else:
	                    queue.append((node.left, level+1))
	                    queue.append((node.right, level+1))

### 运行结果

	Runtime: 692 ms, faster than 88.01% of Python online submissions for Minimum Depth of Binary Tree.
	Memory Usage: 92.4 MB, less than 78.23% of Python online submissions for Minimum Depth of Binary Tree.

原题链接：https://leetcode.com/problems/minimum-depth-of-binary-tree/



您的支持是我最大的动力
