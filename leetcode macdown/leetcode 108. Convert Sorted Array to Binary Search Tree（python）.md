leetcode  108. Convert Sorted Array to Binary Search Tree（python）

### 描述



Given an integer array nums where the elements are sorted in ascending order, convert it to a height-balanced binary search tree.

A height-balanced binary tree is a binary tree in which the depth of the two subtrees of every node never differs by more than one.

Example 1:

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4849d89a60ca446382435dd01c62b2e4~tplv-k3u1fbpfcp-zoom-1.image)

	Input: nums = [-10,-3,0,5,9]
	Output: [0,-3,9,-10,null,5]
	Explanation: [0,-10,5,null,-3,null,9] is also accepted:

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2f59692f29d445f8984f7f7e5e144edb~tplv-k3u1fbpfcp-zoom-1.image)

	
Example 2:

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/fd605e6a7bbb4900909baa0e0481a37c~tplv-k3u1fbpfcp-zoom-1.image)

	
	Input: nums = [1,3]
	Output: [3,1]
	Explanation: [1,3] and [3,1] are both a height-balanced BSTs.



Note:

	1 <= nums.length <= 10^4
	-10^4 <= nums[i] <= 10^4
	nums is sorted in a strictly increasing order.



### 解析

根据题意，就是给出了一个整数列表 nums ，其中的元素都是已经经过升序排序的列表。题目要求我们将 nums 中的整数转换成一棵高度平衡的二叉搜索树，返回其根节点。

高度平衡二叉树是一种特殊二叉树，其中每个节点的两个子树的深度相差不超过 1。

思路很简单，之前提到了很多次，直接使用递归解决就行了，只不过要结合题意设计一下结构。这里要把已经排序的列表 nums 转换成一棵高度平衡二叉树，关键在于先找出根节点，所以只需要将 nums[MID] 的值作为根节点，然后对 nums[:MID] 进行左子树的递归，对 nums[MID+1:] 进行右子树的递归，即可得到符合题意的高度平衡二叉树。

### 解答
				
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def sortedArrayToBST(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: TreeNode
	        """
	        if not nums:return None
	        MID = len(nums)//2
	        root = TreeNode(nums[MID])
	        root.left = self.sortedArrayToBST(nums[:MID])
	        root.right = self.sortedArrayToBST(nums[MID+1:])
	        return root
	


            	      
			
### 运行结果

	Runtime: 97 ms, faster than 32.83% of Python online submissions for Convert Sorted Array to Binary Search Tree.
	Memory Usage: 16 MB, less than 92.78% of Python online submissions for Convert Sorted Array to Binary Search Tree.


原题链接：https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/



您的支持是我最大的动力
