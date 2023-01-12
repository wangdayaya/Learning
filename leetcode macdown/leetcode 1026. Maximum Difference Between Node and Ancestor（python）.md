leetcode  1026. Maximum Difference Between Node and Ancestor（python）

### 描述

Given the root of a binary tree, find the maximum value v for which there exist different nodes a and b where v = |a.val - b.val| and a is an ancestor of b.

A node a is an ancestor of b if either: any child of a is equal to b or any child of a is an ancestor of b.



Example 1:

![](https://assets.leetcode.com/uploads/2020/11/09/tmp-tree.jpg)

	Input: root = [8,3,10,1,6,null,14,null,null,4,7,13]
	Output: 7
	Explanation: We have various ancestor-node differences, some of which are given below :
	|8 - 3| = 5
	|3 - 7| = 4
	|8 - 1| = 7
	|10 - 13| = 3
	Among all possible differences, the maximum value of 7 is obtained by |8 - 1| = 7.


	
Example 2:

![](https://assets.leetcode.com/uploads/2020/11/09/tmp-tree-1.jpg)

	Input: root = [1,null,2,null,0,3]
	Output: 3



Note:


	The number of nodes in the tree is in the range [2, 5000].
	0 <= Node.val <= 10^5

### 解析


根据题意，就是给出了一个二叉树，然后计算祖先节点和其任意子节点的绝对差值最大是多少。思路比较简单，其实对于每个叶子结点来说，最大差值就是其到根节点的最大值和最小值的差值：

* 定义递归函数 dfs ，用来计算根节点到当前节点的最大差值，其中有三个参数，第一个参数是当前节点 root ，第二个和第三个参数记录的是根节点到当前结点路径上的最小值 mn 和最大值 mx ，
* 当遇到叶子结点就计算 mx-mn ，否则就在左右两个子树不断调用 dfs 找出最大值
* 递归结束即可得到答案


### 解答
				
	
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def maxAncestorDiff(self, root):
	        """
	        :type root: TreeNode
	        :rtype: int
	        """
	        def dfs(root, mn, mx):
	            if not root: return mx-mn
	            mn = min(mn, root.val)
	            mx = max(mx, root.val)
	            left = dfs(root.left, mn, mx)
	            right = dfs(root.right, mn, mx)
	            return max(left, right)
	        return dfs(root, root.val, root.val)
            	      
			
### 运行结果

	Runtime: 28 ms, faster than 72.18% of Python online submissions for Maximum Difference Between Node and Ancestor.
	Memory Usage: 19.9 MB, less than 42.86% of Python online submissions for Maximum Difference Between Node and Ancestor.

### 解析

* 使用 result 变量记录最大的绝对差值
* 定义递归函数 dfs ，用来计算根节点到当前节点的最大差值，其中有三个参数，第一个参数是当前节点 root ，第二个和第三个参数记录的是根节点到当前节点路径上的最小值 mn 和最大值 mx ，在左右子树上递归调用 dfs 更新 result 
* 递归结束，得到的 result 就是答案

### 解答


	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def maxAncestorDiff(self, root):
	        """
	        :type root: TreeNode
	        :rtype: int
	        """
	        self.result = 0
	        def dfs(node, cur_max, cur_min):
	            if not node:
	                return
	            self.result = max(self.result, abs(cur_max-node.val), abs(cur_min-node.val))
	            cur_max = max(cur_max, node.val)
	            cur_min = min(cur_min, node.val)
	            dfs(node.left, cur_max, cur_min)
	            dfs(node.right, cur_max, cur_min)
	
	        dfs(root, root.val, root.val)
	        return self.result
	        
### 运行结果
	      
	Runtime: 32 ms, faster than 50.38% of Python online submissions for Maximum Difference Between Node and Ancestor.
	Memory Usage: 20.3 MB, less than 6.02% of Python online submissions for Maximum Difference Between Node and Ancestor.    
        
        
### 解析

上面的递归可能不好理解，下面设计的递归比较易懂，定义 DFS 返回当前节点为根节点的子树中的最大值和最小值，先找左子树的最小值和最大值，然后经过比较得到当前节点与左子树中某值的最大差值，同时更新结果 result ，然后再找右子树的最小值和最大值，然后经过比较得到当前节点与右子树中某值的最大差值，同时更新结果 result ，返回的是当前子树中的最小值和最大值。

### 解答

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def __init__(self):
	        self.result = 0
	    def maxAncestorDiff(self, root):
	        """
	        :type root: TreeNode
	        :rtype: int
	        """
	        self.dfs(root)
	        return self.result
	    
	    def dfs(self, root):
	        if not root: return 100000, 0
	        mn_l, mx_l = self.dfs(root.left)
	        mn_l = min(root.val, mn_l)
	        mx_l = max(root.val, mx_l)
	        self.result = max(max(self.result, abs(root.val-mx_l)), abs(root.val-mn_l))
	        
	        mn_r, mx_r = self.dfs(root.right)
	        mn_r = min(root.val, mn_r)
	        mx_r = max(root.val, mx_r)
	        self.result = max(max(self.result, abs(root.val-mx_r)), abs(root.val-mn_r))
	        return min(mn_l, mn_r), max(mx_l, mx_r)
	        


### 运行结果

	Runtime: 46 ms, faster than 9.85% of Python online submissions for Maximum Difference Between Node and Ancestor.
	Memory Usage: 19.2 MB, less than 78.79% of Python online submissions for Maximum Difference Between Node and Ancestor.

    
原题链接：https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/



您的支持是我最大的动力
