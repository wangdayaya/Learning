leetcode  124. Binary Tree Maximum Path Sum（python）

### 描述



A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any non-empty path.

Example 1:


![](https://assets.leetcode.com/uploads/2020/10/13/exx1.jpg)

	Input: root = [1,2,3]
	Output: 6
	Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.
	
Example 2:

![](https://assets.leetcode.com/uploads/2020/10/13/exx2.jpg)

	Input: root = [-10,9,20,null,null,15,7]
	Output: 42
	Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.




Note:

	The number of nodes in the tree is in the range [1, 3 * 10^4].
	-1000 <= Node.val <= 1000


### 解析


根据题意，二叉树中的路径是节点序列， 一个节点最多只能在序列中出现一次。 需要注意的是这里的路径可以任意长度，而且不是所有的路径都要通过根节点。路径的总和是路径中节点值的总和。给定二叉树的根，返回任何非空路径的最大路径和。

其实一个路径和的模式肯定是要以一个根结点为拐点，然后分别去找左右两条路径，左路径和尽可能大，同理右路径和尽可能大，这样才能保证找出一个非空路径的最大路径和。定义一个递归函数 DFS ，这里没有用 DFS 递归函数直接求解，而是返回以当前节点为根结点向下没有拐弯一直到底的的最大的路径和，因为这个递归函数会遍历所有的节点，所以在遍历所有节点的同时，可以在递归计算左右两条路径保证和尽可能大的同时，顺便就能更新全局变量 result 。计算出非空路径的最大路径和。递归函数不是重点，重点就是在遍历递归过程中更新 result 。

### 解答
				
	
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def __init__(self):
	        self.result = -float('inf')
	        
	    def maxPathSum(self, root):
	        self.dfs(root)
	        return self.result
	    
	    def dfs(self, root):
	        if not root: 
	            return 0
	        L = self.dfs(root.left)
	        R = self.dfs(root.right)
	        
	        PathSum = root.val
	        if L>=0: PathSum += L
	        if R>=0: PathSum += R
	        self.result = max(self.result, PathSum)
	        
	        if L<=0 and R<=0:
	            return root.val
	        else:
	            return root.val + max(L, R)
	            
            	      
			
### 运行结果

	Runtime: 84 ms, faster than 51.00% of Python online submissions for Binary Tree Maximum Path Sum.
	Memory Usage: 25.5 MB, less than 93.75% of Python online submissions for Binary Tree Maximum Path Sum.



原题链接：https://leetcode.com/problems/binary-tree-maximum-path-sum/



会当凌绝顶，一览众山小
