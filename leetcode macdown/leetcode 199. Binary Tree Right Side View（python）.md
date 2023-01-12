leetcode  199. Binary Tree Right Side View（python）




### 描述

Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.



Example 1:

![](https://assets.leetcode.com/uploads/2021/02/14/tree.jpg)

	Input: root = [1,2,3,null,5,null,4]
	Output: [1,3,4]

	
Example 2:

	Input: root = [1,null,3]
	Output: [1,3]


Example 3:


	Input: root = []
	Output: []


Note:

	The number of nodes in the tree is in the range [0, 100].
	-100 <= Node.val <= 100


### 解析

根据题意，给定二叉树的根 root ，想象你站在它的右边，返回你可以看到从上到下能看到的最右边的节点的值。

因为我们要找到每一层的最右边的节点值，所以我们要按照层进行遍历节点，将所有最右边的值放入结果 result 中皆可，其实这道题考察的就是使用使用 BFS 进行层序遍历。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def rightSideView(self, root):
	        """
	        :type root: TreeNode
	        :rtype: List[int]
	        """
	        if not root:
	            return []
	        q = [root]
	        result = [root.val]
	        while q:
	            l =  len(q)
	            for i in range(l):
	                node = q.pop(0)
	                if node.left:
	                    q.append(node.left)
	                if node.right:
	                    q.append(node.right)
	            if q:
	                result.append(q[-1].val)
	        return result
	


### 运行结果


	Runtime: 24 ms, faster than 70.72% of Python online submissions for Binary Tree Right Side View.
	Memory Usage: 13.4 MB, less than 52.08% of Python online submissions for Binary Tree Right Side View.
### 原题链接

https://leetcode.com/problems/binary-tree-right-side-view/


您的支持是我最大的动力
