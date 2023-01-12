leetcode  1325. Delete Leaves With a Given Value（python）

### 描述

Given a binary tree root and an integer target, delete all the leaf nodes with value target.

Note that once you delete a leaf node with value target, if it's parent node becomes a leaf node and has the value target, it should also be deleted (you need to continue doing that until you can't).



Example 1:

![](https://assets.leetcode.com/uploads/2020/01/09/sample_1_1684.png)

	Input: root = [1,2,3,2,null,2,4], target = 2
	Output: [1,null,3,null,4]
	Explanation: Leaf nodes in green with value (target = 2) are removed (Picture in left). 
	After removing, new nodes become leaf nodes with value (target = 2) (Picture in center).

	
Example 2:


![](https://assets.leetcode.com/uploads/2020/01/09/sample_2_1684.png)

	Input: root = [1,3,3,3,2], target = 3
	Output: [1,3,null,null,2]

Example 3:

![](https://assets.leetcode.com/uploads/2020/01/15/sample_3_1684.png)

	Input: root = [1,2,null,2,null,2], target = 2
	Output: [1]
	Explanation: Leaf nodes in green with value (target = 2) are removed at each step.


	
Example 4:


	Input: root = [1,1,1], target = 1
	Output: []
	
Example 5:


	Input: root = [1,2,3], target = 1
	Output: [1,2,3]

Note:

	1 <= target <= 1000
	The given binary tree will have between 1 and 3000 nodes.
	Each node's value is between [1, 1000].


### 解析

根据题意，就是给出了一个二叉树，让我们删除值为 target 的叶子结点，注意，在删除叶子结点之后可能父节点也称为了叶子结点，如果值为 target 也应该删除，不断重复此类操作直到不能进行下去，返回新树的索引。思路比较简单：

* 就是使用了递归思想，当前节点为空就直接返回 None ；
* 否则判断左子树是否为空；判断右子树是否为空；
* 当左右子树都为空且值为 target 的时候，说明其不该存在，就直接返回 None ；否则说明其应该存在，返回 root ；
* 递归节点返回的 root 即为新树的引用


### 解答
				

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def removeLeafNodes(self, root, target):
	        """
	        :type root: TreeNode
	        :type target: int
	        :rtype: TreeNode
	        """
	        if not root:
	            return None
	        root.left = self.removeLeafNodes(root.left, target)
	        root.right = self.removeLeafNodes(root.right, target)
	        if not root.left and not root.right and root.val == target:
	            return None
	        return root
            	      
			
### 运行结果


	Runtime: 44 ms, faster than 52.54% of Python online submissions for Delete Leaves With a Given Value.
	Memory Usage: 14.4 MB, less than 17.80% of Python online submissions for Delete Leaves With a Given Value.

原题链接：https://leetcode.com/problems/delete-leaves-with-a-given-value/



您的支持是我最大的动力
