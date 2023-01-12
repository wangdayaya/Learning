leetcode  979. Distribute Coins in Binary Tree（python）

### 描述


You are given the root of a binary tree with n nodes where each node in the tree has node.val coins. There are n coins in total throughout the whole tree.

In one move, we may choose two adjacent nodes and move one coin from one node to another. A move may be from parent to child, or from child to parent.

Return the minimum number of moves required to make every node have exactly one coin.


Example 1:


![](https://assets.leetcode.com/uploads/2019/01/18/tree1.png)

	Input: root = [3,0,0]
	Output: 2
	Explanation: From the root of the tree, we move one coin to its left child, and one coin to its right child.
	
Example 2:


![](https://assets.leetcode.com/uploads/2019/01/18/tree2.png)

	Input: root = [0,3,0]
	Output: 3
	Explanation: From the left child of the root, we move two coins to the root [taking two moves]. Then, we move one coin from the root of the tree to the right child.


Example 3:


![](https://assets.leetcode.com/uploads/2019/01/18/tree3.png)

	Input: root = [1,0,2]
	Output: 2
	
Example 4:

![](https://assets.leetcode.com/uploads/2019/01/18/tree4.png)

	Input: root = [1,0,0,null,3]
	Output: 4



Note:

	The number of nodes in the tree is n.
	1 <= n <= 100
	0 <= Node.val <= n
	The sum of all Node.val is n.


### 解析


根据题意，就是给出了一个 n 个节点的二叉树，并且给出了 n 个金币，但是金币的位置是随机的，不知道在哪些节点上，我们现在有一种操作，每次可以将一个金币从 child 节点移动到 parent 节点，或者从 parent 节点移动到 child 节点，结果就是求最少有多少次操作，可以将每个节点上都至少有一个金币。

其实说白了有 n 个节点同时有 n 个金币，就是每个将金币分散到每个节点上，每个节点得到一个。这是一个二叉树的平衡问题，最终的操作次数就是左子树和右子树的不平衡总和，以例子 3 为例，左子树的不平衡状态为 1 ，相当于缺一个金币，需要从 root 拿一个金币，右子树的不平衡状态为 1 ，相当于多出来一个金币需要往 root 上交一个，所以总的操作次数就是左右子树各自不平衡金币个数绝对值之和。

我们使用 result 来计数总共的操作次数，每次返回的不平衡结果=当前的金币+左子树不平衡的金币绝对值+右子树不平衡的金币绝对值-1

### 解答
				
	
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def distributeCoins(self, root):
	        """
	        :type root: TreeNode
	        :rtype: int
	        """
	        self.result = 0
	
	        def dfs(node):
	            if not node: return 0
	            L, R = dfs(node.left), dfs(node.right)
	            self.result += abs(L) + abs(R)
	            return node.val + L + R - 1
	
	        dfs(root)
	        return self.result
	        
            	      
			
### 运行结果

	Runtime: 49 ms, faster than 5.19% of Python online submissions for Distribute Coins in Binary Tree.
	Memory Usage: 13.4 MB, less than 92.21% of Python online submissions for Distribute Coins in Binary Tree.



原题链接：https://leetcode.com/problems/distribute-coins-in-binary-tree/



您的支持是我最大的动力
