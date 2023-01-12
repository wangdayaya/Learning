leetcode 968. Binary Tree Cameras （python）




### 描述

You are given the root of a binary tree. We install cameras on the tree nodes where each camera at a node can monitor its parent, itself, and its immediate children.

Return the minimum number of cameras needed to monitor all nodes of the tree.



Example 1:


![](https://assets.leetcode.com/uploads/2018/12/29/bst_cameras_01.png)

	Input: root = [0,0,null,0,0]
	Output: 1
	Explanation: One camera is enough to monitor all nodes if placed as shown.
	
Example 2:


![](https://assets.leetcode.com/uploads/2018/12/29/bst_cameras_02.png)
	
	Input: root = [0,0,null,0,null,0,null,null,0]
	Output: 2
	Explanation: At least two cameras are needed to monitor all nodes of the tree. The above image shows one of the valid configurations of camera placement.





Note:

	The number of nodes in the tree is in the range [1, 1000].
	Node.val == 0


### 解析

根据题意，给定一个二叉树的根 root 。 我们在树节点上安装摄像头，节点上的每个摄像头都可以监控其父节点、自身及其直接子节点。返回监视树的所有节点所需的最小摄像机数。

记住一点，遇到二叉树的问题，我们就用递归的形式解题。这道题里面还隐藏了贪心的思想，我们先定义三种状态：

*         0 表示没有相机，也没有相机覆盖
*         1 表示装了相机
*         2 表示被相机覆盖，但是没有装相机

这三种状态就包含了二叉树中出现的所有情况，按照贪心的思想，我们尽可能地从下往上让父节点去安装摄像头，因为这样摄像头才能覆盖更大的范围，我们使用摄像头的数量也会最少。具体的递归解决思路都在注释里面，结合代码看会一目了然。

时间复杂度为 O(N) ，空间复杂度为 O(K) ，K 为树的深度，也就是递归栈的深度。

### 解答
				

	class Solution(object):
	    def __init__(self):
	        self.result = 0
	
	    def minCameraCover(self, root):
	        if not root.left and not root.right: return 1
	
	        def dfs(node):
	            if not node: return 2
	            L = dfs(node.left)
	            R = dfs(node.right)
	            # 如果左子树和右子树都有被相机覆盖，那么当前节点啥都不干，等待其父节点装相机，这样能减少装相机的数量
	            if L == 2 and R == 2:
	                return 0
	            # 至少有一个节点没有被覆盖，当前节点要安装照相机
	            elif L == 0 or R == 0:
	                self.result += 1
	                return 1
	            # 如果左子树或者有子树有相机，当前节点被相机覆盖，所以不需要安装照相机
	            elif L == 1 or R == 1:
	                return 2
	
	        if dfs(root) == 0:
	            self.result += 1
	        return self.result
            	      
			
### 运行结果


	Runtime: 27 ms, faster than 91.23% of Python online submissions for Binary Tree Cameras.
	Memory Usage: 14 MB, less than 22.81% of Python online submissions for Binary Tree Cameras.

### 原题链接


https://leetcode.com/problems/binary-tree-cameras/


您的支持是我最大的动力
