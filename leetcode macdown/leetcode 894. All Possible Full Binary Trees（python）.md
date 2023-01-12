leetcode 894. All Possible Full Binary Trees （python）

### 描述


Given an integer n, return a list of all possible full binary trees with n nodes. Each node of each tree in the answer must have Node.val == 0.

Each element of the answer is the root node of one possible tree. You may return the final list of trees in any order.

A full binary tree is a binary tree where each node has exactly 0 or 2 children.




Example 1:

![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/08/22/fivetrees.png)

	Input: n = 7
	Output: [[0,0,0,null,null,0,0,null,null,0,0],[0,0,0,null,null,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,null,null,null,null,0,0],[0,0,0,0,0,null,null,0,0]]

	
Example 2:

	Input: n = 3
	Output: [[0,0,0]]



Note:

	1 <= n <= 20


### 解析

根据题意，给出了一个整数 n ，让我们返回所有可能的满二叉树（要求所有的结点都有 0 个或者 2 个叶子结点的树），并且所有的节点的值都为  0 ，将所有可能的满二叉树的根都放入最后的结果列表中并返回，二叉树的顺序可以是任意的。

其实碰到树类型的数据结构，大概率是用到了递归了，这道题也是用到了递归。其实自己列举几个列子就能发现，满二叉树得要求 n 为正奇数才行，否则直接返回空列表就行了，因为偶数个数的节点无法构成满二叉树。

如果 n 为 0 则直接返回空列表
如果 n 为  1 则直接返回只有一个节点值为 0 的根节点列表
如果 n 为 3 则只能构成一种满二叉树，直接返回根节点列表
如果 n 为 5 则会出现两种满二叉树，左右子树节点个数可能是左 1 右 3 ，也可能是左 3 右 1 ，直接返回两个树的根节点列表
如果 n 为 7 则会出现五种二叉树，左右子树节点个数可能是左 1 右 5 ，也可能是左 5 右 1 ，左 3 右 3 ，直接返回五个树的根节点列表
...
通过找规律可以看出构成的结果列表中的树都需要对节点进行排列组合，需要三层循环来完成，第一层来控制左子树的节点个数 l ，第二层遍历 l 个左根节点，第三层遍历 N-1-l 个右根节点。


### 解答
				

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def allPossibleFBT(self, n):
	        """
	        :type n: int
	        :rtype: List[TreeNode]
	        """
	        if n%2==0: return []
	        if n == 1: return [TreeNode(0)]
	        N = n - 1
	        res = []
	        for l in range(1, N, 2):
	            for left in self.allPossibleFBT(l):
	                for right in self.allPossibleFBT(N - l):
	                    node = TreeNode(0)
	                    node.left = left
	                    node.right = right
	                    res.append(node)
	        return res
	        
	        
            	      
			
### 运行结果

	Runtime: 296 ms, faster than 24.35% of Python online submissions for All Possible Full Binary Trees.
	Memory Usage: 42.5 MB, less than 7.25% of Python online submissions for All Possible Full Binary Trees.

### 解析

另外一种解法是在官网上看某大神的解答，用到的是动态规划，思路是真的秒的不行，而且从结果看，速度和所占内存都有很大的提升，不佩服不行啊。



### 解答


	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def allPossibleFBT(self, n):
	        """
	        :type n: int
	        :rtype: List[TreeNode]
	        """
	        dp = [[] for _ in range(n + 1)]
	        dp[1] = [TreeNode()]
	        for i in range(1, n + 1, 2):
	            for j in range(1, i, 2):
	                for tree1 in dp[j]:
	                    for tree2 in dp[i - j - 1]:
	                        dp[i].append(TreeNode(0, tree1, tree2))
	        return dp[-1]
        
        
### 运行结果

	Runtime: 184 ms, faster than 77.72% of Python online submissions for All Possible Full Binary Trees.
	Memory Usage: 17.2 MB, less than 62.69% of Python online submissions for All Possible Full Binary Trees.
	
	
原题链接：https://leetcode.com/problems/all-possible-full-binary-trees/



您的支持是我最大的动力
