leetcode  938. Range Sum of BST（python）

### 描述


Given the root node of a binary search tree and two integers low and high, return the sum of values of all nodes with a value in the inclusive range [low, high].


Example 1:

![](https://assets.leetcode.com/uploads/2020/11/05/bst1.jpg)

	Input: root = [10,5,15,3,7,null,18], low = 7, high = 15
	Output: 32
	Explanation: Nodes 7, 10, and 15 are in the range [7, 15]. 7 + 10 + 15 = 32.

	
Example 2:

![](https://assets.leetcode.com/uploads/2020/11/05/bst2.jpg)

	Input: root = [10,5,15,3,7,13,18,1,null,6], low = 6, high = 10
	Output: 23
	Explanation: Nodes 6, 7, and 10 are in the range [6, 10]. 6 + 7 + 10 = 23.


Note:

	The number of nodes in the tree is in the range [1, 2 * 10^4].
	1 <= Node.val <= 10^5
	1 <= low <= high <= 10^5
	All Node.val are unique.


### 解析


根据题意，就是给出了一颗二叉树，和一个范围 [low, high] ，让我们求在这个范围内的节点的值的和是多少，思路比较简单，就是用递归的思想，判断每个节点的值如果在 [low, high] 中就累加起来，否则不去管他，最后递归的结果就是答案。

### 解答
				

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def rangeSumBST(self, root, low, high):
	        """
	        :type root: TreeNode
	        :type low: int
	        :type high: int
	        :rtype: int
	        """
	        if not root: return 0
	        def dfs(root):
	            if not root:
	                return 0
	            currentValue =  root.val if low <= root.val <= high else 0
	            return currentValue+dfs(root.left)+dfs(root.right)
	        return dfs(root)
            	      
			
### 运行结果

	Runtime: 296 ms, faster than 47.48% of Python online submissions for Range Sum of BST.
	Memory Usage: 29.7 MB, less than 41.77% of Python online submissions for Range Sum of BST.

### 解析


上面的解法直接通过递归遍历了所有的节点，其实我们可以根据二叉树左子树小于根结点，右子树大雨根结点来减少计算量。仍然使用递归的思想。明显运行时间缩小了。

### 解答
				
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def rangeSumBST(self, root, low, high):
	        """
	        :type root: TreeNode
	        :type low: int
	        :type high: int
	        :rtype: int
	        """
	        if not root: return 0
	        if root.val>high:
	            return self.rangeSumBST(root.left, low, high)
	        if root.val<low:
	            return self.rangeSumBST(root.right, low, high)
	        return root.val + self.rangeSumBST(root.left, low, high) +  self.rangeSumBST(root.right, low, high)

            	      
			
### 运行结果

	Runtime: 260 ms, faster than 61.37% of Python online submissions for Range Sum of BST.
	Memory Usage: 29.9 MB, less than 17.45% of Python online submissions for Range Sum of BST.
### 解析
同样的思路，不同的写法
### 解答
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def rangeSumBST(self, root, low, high):
	        """
	        :type root: TreeNode
	        :type low: int
	        :type high: int
	        :rtype: int
	        """
	        def dfs(root):
	            if not root: return 0
	            if root.val>high:
	                return dfs(root.left)
	            if root.val<low:
	                return dfs(root.right)
	            return root.val + dfs(root.left) + dfs(root.right)
	        return dfs(root)
	        
### 运行结果	        
    Runtime: 248 ms, faster than 79.16% of Python online submissions for Range Sum of BST.
	Memory Usage: 29.6 MB, less than 69.89% of Python online submissions for Range Sum of BST.   

原题链接：https://leetcode.com/problems/range-sum-of-bst/



您的支持是我最大的动力
