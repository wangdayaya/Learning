leetcode  1448. Count Good Nodes in Binary Tree（python）

### 描述


Given a binary tree root, a node X in the tree is named good if in the path from root to X there are no nodes with a value greater than X.

Return the number of good nodes in the binary tree.




Example 1:

![](https://assets.leetcode.com/uploads/2020/04/02/test_sample_1.png)

	Input: root = [3,1,4,3,null,1,5]
	Output: 4
	Explanation: Nodes in blue are good.
	Root Node (3) is always a good node.
	Node 4 -> (3,4) is the maximum value in the path starting from the root.
	Node 5 -> (3,4,5) is the maximum value in the path
	Node 3 -> (3,1,3) is the maximum value in the path.

	
Example 2:

![](https://assets.leetcode.com/uploads/2020/04/02/test_sample_2.png)

	Input: root = [3,3,null,4,2]
	Output: 3
	Explanation: Node 2 -> (3, 3, 2) is not good, because "3" is higher than it.

Example 3:

	Input: root = [1]
	Output: 1
	Explanation: Root is considered as good.

	


Note:

	The number of nodes in the binary tree is in the range [1, 10^5].
	Each node's value is between [-10^4, 10^4].


### 解析


根据题意，就是给出了一个二叉树 root ，找出这棵树中间有多少个节点是 good 的，题目中对“ good 节点”的定义为该节点到根结点的路径上的节点的值都小于等于该节点。思路比较简单，直接用 DFS ，在递归的时候传入当前路径中的最大值，然后和当前值进行比较，如果符合题意，则结果 result 加一，递归结束的 result 即为答案。

### 解答
				
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def __init__(self):
	        self.result = 0
	
	    def goodNodes(self, root):
	        """
	        :type root: TreeNode
	        :rtype: int
	        """
	        if not root:
	            return 0
	
	        def dfs(root, maxValue):
	            if not root:
	                return
	            maxValue = max(maxValue, root.val)
	            if root.val >= maxValue:
	                self.result += 1
	            dfs(root.left, maxValue)
	            dfs(root.right, maxValue)
	            
	        dfs(root, root.val)
	        return self.result

            	      
			
### 运行结果


	Runtime: 300 ms, faster than 53.08% of Python online submissions for Count Good Nodes in Binary Tree.
	Memory Usage: 49.5 MB, less than 9.80% of Python online submissions for Count Good Nodes in Binary Tree.

### 解析


不使用上面的全局变量 self.result ，直接用递归进行计算结果。

### 解答


	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def goodNodes(self, root):
	        """
	        :type root: TreeNode
	        :rtype: int
	        """
	        if not root or (not root.left and not root.right):
	            return 1
	
	        def dfs(root, maxValue):
	            if not root:
	                return 0
	            maxValue = max(maxValue, root.val)
	            count = 1 if root.val >= maxValue else 0
	            return count + dfs(root.left, maxValue) + dfs(root.right, maxValue)
	
	        return dfs(root, root.val)
### 运行结果	        

	Runtime: 308 ms, faster than 41.65% of Python online submissions for Count Good Nodes in Binary Tree.
	Memory Usage: 49.4 MB, less than 42.93% of Python online submissions for Count Good Nodes in Binary Tree.
   
原题链接：https://leetcode.com/problems/count-good-nodes-in-binary-tree/



您的支持是我最大的动力
