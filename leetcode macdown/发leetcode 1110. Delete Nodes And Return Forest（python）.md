leetcode  1110. Delete Nodes And Return Forest（python）

### 描述


Given the root of a binary tree, each node in the tree has a distinct value.

After deleting all nodes with a value in to_delete, we are left with a forest (a disjoint union of trees).

Return the roots of the trees in the remaining forest. You may return the result in any order.

 


Example 1:

![](https://assets.leetcode.com/uploads/2019/07/01/screen-shot-2019-07-01-at-53836-pm.png)

	Input: root = [1,2,3,4,5,6,7], to_delete = [3,5]
	Output: [[1,2,null,4],[6],[7]]

	
Example 2:


	Input: root = [1,2,4,null,3], to_delete = [3]
	Output: [[1,2,4]]





Note:

	The number of nodes in the given tree is at most 1000.
	Each node has a distinct value between 1 and 1000.
	to_delete.length <= 1000
	to_delete contains distinct values between 1 and 1000.


### 解析

根据题意，就是给出了一个二叉树，并且给出了一个需要删除的节点值列表 to\_delete ，题目要求我们将节点值出现在 to_delete 中的节点都删去，并且将删除之后形成的多个森林的引用都放入列表中。最后的列表结果是不需要排序的。

递归函数的操作是比较繁琐的，因为假如当前节点是需要被删除的，那么需要将其形成的左右非空两个子树加入到结果列表中，并且要将当前节点置空。这里的节点置空操作是需要注意的，因为结点置空不是直接 root=None ， 而是需要将其父节点的 left 引用或者 right 引用置空才对。另外需要注意的是别忘了将根节点也加入进结果列表中。


### 解答
				
	
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def delNodes(self, root, to_delete):
	        """
	        :type root: TreeNode
	        :type to_delete: List[int]
	        :rtype: List[TreeNode]
	        """
	        result = []
	        def dfs(root):
	            if not root: return None
	            root.left = dfs(root.left)
	            root.right = dfs(root.right)
	            if root.val in to_delete:
	                if root.left:
	                    result.append(root.left)
	                if root.right:
	                    result.append(root.right)
	                return None
	            return root
	        dfs(root)
	        if root.val not in to_delete:
	            result.append(root)
	        return result
            	      
			
### 运行结果

	Runtime: 110 ms, faster than 5.12% of Python online submissions for Delete Nodes And Return Forest.
	Memory Usage: 13.9 MB, less than 92.47% of Python online submissions for Delete Nodes And Return Forest.


原题链接：https://leetcode.com/problems/delete-nodes-and-return-forest/



您的支持是我最大的动力
