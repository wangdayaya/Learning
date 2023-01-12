leetcode 543. Diameter of Binary Tree （python）
### 每日经典

《观沧海》 ——曹操(三国) 

东临碣石，以观沧海。
水何澹澹，山岛竦峙。
树木丛生，百草丰茂。
秋风萧瑟，洪波涌起。
日月之行，若出其中；
星汉灿烂，若出其里。
幸甚至哉，歌以咏志。

### 描述

Given the root of a binary tree, return the length of the diameter of the tree.

The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

The length of a path between two nodes is represented by the number of edges between them.



Example 1:

![](https://assets.leetcode.com/uploads/2021/03/06/diamtree.jpg)

	Input: root = [1,2,3,4,5]
	Output: 3
	Explanation: 3 is the length of the path [4,2,1,3] or [5,2,1,3].

	
Example 2:
	
	Input: root = [1,2]
	Output: 1




Note:

	The number of nodes in the tree is in the range [1, 10^4].
	-100 <= Node.val <= 100


### 解析

根据题意，给定二叉树的根 root ，返回树的直径长度。二叉树的直径是树中任意两个节点之间最长路径的长度。 此路径可能通过根结点也可能不会通过根节点。两个节点之间的路径长度由它们之间的边数表示。

其实这道题还是和 【687. Longest Univalue Path】的思路是一样的，只不过这道题更加简单，因为以某个节点为根结点，其直径的长度肯定是由左边尽可能向下延伸到叶子节点的长度 L ，和右边尽可能向下延伸到叶子节点的长度 R 两部分组成的，所以我们定义了一个递归函数 DFS ，返回以当点节点为根结点到最远叶子节点的路径上的节点个数。这样在节点左子树调用 DFS 得到 l ，和在节点右子树调用 DFS 得到 r ，使用 l+r+1 更新最大的 result 即可。使用 DFS 最后得到的 result 只是最大的节点数，还需要减一才是最终题目定义的直径长度。


### 解答
				
	
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	        
	class Solution(object):
	    def __init__(self):
	        self.result = 0
	        
	    def diameterOfBinaryTree(self, root):
	        """
	        :type root: TreeNode
	        :rtype: int
	        """
	        self.dfs(root)
	        return self.result-1
	    
	    def dfs(self, root):
	        if not root: return 0
	        L = self.dfs(root.left)
	        R = self.dfs(root.right)
	        self.result = max(self.result, L+R+1)
	        return max(L, R) + 1
            	      
			
### 运行结果

	Runtime: 28 ms, faster than 92.48% of Python online submissions for Diameter of Binary Tree.
	Memory Usage: 16 MB, less than 86.00% of Python online submissions for Diameter of Binary Tree.


原题链接：https://leetcode.com/problems/diameter-of-binary-tree/



您的支持是我最大的动力
