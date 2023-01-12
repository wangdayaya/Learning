leetcode  669. Trim a Binary Search Tree（python）




### 描述


Given the root of a binary search tree and the lowest and highest boundaries as low and high, trim the tree so that all its elements lies in [low, high]. Trimming the tree should not change the relative structure of the elements that will remain in the tree (i.e., any node's descendant should remain a descendant). It can be proven that there is a unique answer.

Return the root of the trimmed binary search tree. Note that the root may change depending on the given bounds.


Example 1:

![](https://assets.leetcode.com/uploads/2020/09/09/trim1.jpg)

	Input: root = [1,0,2], low = 1, high = 2
	Output: [1,null,2]

	
Example 2:

![](https://assets.leetcode.com/uploads/2020/09/09/trim2.jpg)

	Input: root = [3,0,4,null,2,null,null,1], low = 1, high = 3
	Output: [3,2,null,1]



Note:

	The number of nodes in the tree in the range [1, 10^4].
	0 <= Node.val <= 10^4
	The value of each node in the tree is unique.
	root is guaranteed to be a valid binary search tree.
	0 <= low <= high <= 10^4


### 解析


根据题意，给定二叉搜索树的根 root 以及最低和最高边界 low 和 high ，然后修剪树使其所有元素位于 [low, high] 中，不在这个范围的元素都去掉，但是题目要求修剪树的时候要保证任何节点的后代都应该始终保持在其后代的位置，也就是相对位置不能发生变化。可以证明有一个唯一的答案。返回修剪后的二叉搜索树的根 root 。

遇到树的数据结构类型的题目，直接 DFS 梭哈，这道题也不例外。

这道题的难点就是再裁减树中不在范围的点的过程中，我们要能够保证子树与父节点的相对位置保持不变，而每一个节点的裁剪逻辑都是 trimBST ，所以我们直接递归调用 trimBST 即可：

* 如果 root 为空，直接返回即可
* 如果 root.val 在 [low, high] 中，那么说明该节点是正常的，不用裁剪，我们直接去递归裁剪其左节点和右节点即可
* 如果 root.val 小于 low ，说明其左节点已经没用了，直接去递归裁剪右节点即可
* 如果 root.val 大于 high ，说明其右节点已经没用了，直接去递归裁剪左节点即可
* 递归结束返回 root 即可

时间复杂度为 O(N) ，空间复杂度为 O(1) 。



### 解答
				

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def trimBST(self, root, low, high):
	        """
	        :type root: TreeNode
	        :type low: int
	        :type high: int
	        :rtype: TreeNode
	        """
	        if not root:
	            return 
	        if low <= root.val <= high:
	            root.left = self.trimBST(root.left, low, high)
	            root.right = self.trimBST(root.right, low, high)
	        elif root.val < low:
	            root = self.trimBST(root.right, low, high)
	        elif root.val > high:
	            root = self.trimBST(root.left, low, high)
	        return root
            	      
			
### 运行结果

	Runtime: 46 ms, faster than 70.04% of Python online submissions for Trim a Binary Search Tree.
	Memory Usage: 21.3 MB, less than 13.50% of Python online submissions for Trim a Binary Search Tree.



### 原题链接


https://leetcode.com/problems/trim-a-binary-search-tree/


您的支持是我最大的动力
