leetcode  653. Two Sum IV - Input is a BST（python）




### 描述



Given the root of a Binary Search Tree and a target number k, return true if there exist two elements in the BST such that their sum is equal to the given target.


Example 1:

![](https://assets.leetcode.com/uploads/2020/09/21/sum_tree_1.jpg)

	Input: root = [5,3,6,2,4,null,7], k = 9
	Output: true

	
Example 2:


![](https://assets.leetcode.com/uploads/2020/09/21/sum_tree_2.jpg)

	Input: root = [5,3,6,2,4,null,7], k = 28
	Output: false

 


Note:


* The number of nodes in the tree is in the range [1, 10^4].
* -10^4 <= Node.val <= 10^4
* root is guaranteed to be a valid binary search tree.
* -10^5 <= k <= 10^5

### 解析

根据题意，给定二叉搜索树的 root 和目标数字 k ，如果 BST 中存在两个元素，使得它们的总和等于给定的目标 k ，则返回 True ，否则返回 False 。

这道题其实相当简单了，考查的就是元素的遍历，具体方法使用的肯定是 dfs 递归 ，我们使用递归选择一种节点遍历的顺序，将所有的节点的值都存在于一个数组中 values ，这里需要注意的是树中肯定不会出现两个相同的节点值，然后我们使用双指针，从两侧往中间移动，如果指针指向的两个值的和等于 k 那么直接返回 True ，如果遍历结束那么直接返回 False 。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。



### 解答

	class Solution(object):
	    def findTarget(self, root, k):
	        """
	        :type root: TreeNode
	        :type k: int
	        :rtype: bool
	        """
	        self.values = []
	        def dfs(root):
	            if not root:
	                return 
	            dfs(root.left)
	            self.values.append(root.val)
	            dfs(root.right)
	        dfs(root)
	        left, right = 0, len(self.values) - 1
	        while left < right:
	            total = self.values[left] + self.values[right]
	            if total == k:
	                return True
	            if total < k:
	                left += 1
	            else:
	                right -= 1
	        return False

### 运行结果

	Runtime: 167 ms, faster than 23.66% of Python online submissions for Two Sum IV - Input is a BST.
	Memory Usage: 20 MB, less than 21.43% of Python online submissions for Two Sum IV - Input is a BST.
	
### 解析
或者不用像上面那么麻烦，可以边递归，边进行条件判断，我们先预先初始化一个数组 L ，使用 dfs 递归每个根节点，如果 k-root.val 存在于 L 中，那么直接返回 True 即可，否则继续在左右子树中进行递归判断。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def __init__(self):
	        self.L = []
	    def findTarget(self, root, k):
	        """
	        :type root: TreeNode
	        :type k: int
	        :rtype: bool
	        """
	        if not root:
	            return False
	        if k-root.val in self.L:
	            return True
	        self.L.append(root.val)
	        return self.findTarget(root.left, k) or self.findTarget(root.right, k)
### 运行结果
	Runtime: 105 ms, faster than 73.22% of Python online submissions for Two Sum IV - Input is a BST.
	Memory Usage: 18.3 MB, less than 66.52% of Python online submissions for Two Sum IV - Input is a BST.


### 原题链接

https://leetcode.com/problems/two-sum-iv-input-is-a-bst/


您的支持是我最大的动力
