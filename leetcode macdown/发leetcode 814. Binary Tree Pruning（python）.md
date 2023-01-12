leetcode  814. Binary Tree Pruning（python）

### 描述


Given the root of a binary tree, return the same tree where every subtree (of the given tree) not containing a 1 has been removed.

A subtree of a node node is node plus every node that is a descendant of node.




Example 1:

![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/04/06/1028_2.png)

	Input: root = [1,null,0,0,1]
	Output: [1,null,0,null,1]
	Explanation: 
	Only the red nodes satisfy the property "every subtree not containing a 1".
	The diagram on the right represents the answer.

	
Example 2:

![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/04/06/1028_1.png)

	Input: root = [1,0,1,0,0,0,1]
	Output: [1,null,1,null,1]



Example 3:


![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/04/05/1028.png)

	Input: root = [1,1,0,1,1,0,1,0]
	Output: [1,1,0,1,1,null,1]
	


Note:

	
	The number of nodes in the tree is in the range [1, 200].
	Node.val is either 0 or 1.

### 解析

根据题意，就是给出了一个二叉树 root ，然后让我们进行树的剪枝，对不包含 1 的子树进行移除，返回移除了节点后的树的索引。思路比较简单：

就是使用的 DFS ，判断最底层的节点的左子树是否为空，右子树是否为空，如果左右子树都为空且值为 0 的节点，那么将其变为空。然后向上回溯，上层的节点通过同样的方法也能变为空，知道最后将所有节点从下到上都判断一次，然后返回新树的根节点。


### 解答
				
	
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def pruneTree(self, root):
	        """
	        :type root: TreeNode
	        :rtype: TreeNode
	        """
	        def dfs(root):
	            if not root:
	                return None
	            root.left = dfs(root.left)
	            root.right = dfs(root.right)
	            if root.val == 0 and not root.left and not root.right:
	                return None
	            return root
	        return dfs(root)
            	      
			
### 运行结果

	Runtime: 20 ms, faster than 65.79% of Python online submissions for Binary Tree Pruning.
	Memory Usage: 13.6 MB, less than 28.57% of Python online submissions for Binary Tree Pruning.

### 解析

习惯了写函数 dfs ，其实不用写也可以，直接把 pruneTree 当作递归函数使用即可。
### 解答


	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def pruneTree(self, root):
	        """
	        :type root: TreeNode
	        :rtype: TreeNode
	        """
	        if not root:
	            return None
	        root.left = self.pruneTree(root.left)
	        root.right = self.pruneTree(root.right)
	        if root.val == 0 and not root.left and not root.right:
	            return None
	        return root
	
### 运行结果
	
	Runtime: 20 ms, faster than 65.79% of Python online submissions for Binary Tree Pruning.
	Memory Usage: 13.4 MB, less than 54.52% of Python online submissions for Binary Tree Pruning.

原题链接：https://leetcode.com/problems/binary-tree-pruning/



您的支持是我最大的动力
