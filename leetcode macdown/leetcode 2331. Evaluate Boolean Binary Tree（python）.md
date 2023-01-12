leetcode 2331. Evaluate Boolean Binary Tree （python）




### 描述



You are given the root of a full binary tree with the following properties:

* Leaf nodes have either the value 0 or 1, where 0 represents False and 1 represents True.
* Non-leaf nodes have either the value 2 or 3, where 2 represents the boolean OR and 3 represents the boolean AND.

The evaluation of a node is as follows:

* If the node is a leaf node, the evaluation is the value of the node, i.e. True or False.
* Otherwise, evaluate the node's two children and apply the boolean operation of its value with the children's evaluations.

Return the boolean result of evaluating the root node. A full binary tree is a binary tree where each node has either 0 or 2 children. A leaf node is a node that has zero children.

Example 1:


![](https://assets.leetcode.com/uploads/2022/05/16/example1drawio1.png)

	Input: root = [2,1,3,null,null,0,1]
	Output: true
	Explanation: The above diagram illustrates the evaluation process.
	The AND node evaluates to False AND True = False.
	The OR node evaluates to True OR False = True.
	The root node evaluates to True, so we return true.
	
Example 2:

	Input: root = [0]
	Output: false
	Explanation: The root node is a leaf node and it evaluates to false, so we return false.





Note:

	The number of nodes in the tree is in the range [1, 1000].
	0 <= Node.val <= 3
	Every node has either 0 or 2 children.
	Leaf nodes have a value of 0 or 1.
	Non-leaf nodes have a value of 2 or 3.


### 解析

根据题意，给定以下属性的完整二叉树的根：

* 叶节点的值为 0 或 1，其中 0 代表 False，1 代表 True 。
* 非叶节点的值为 2 或 3，其中 2 表示 OR ，3 表示 AND 。

一个树的评价方式如下：

* 如果节点是叶节点，则评估是节点的值，即 True 或 False 
* 否则使用布尔运算评估节点的两个子节点的值

返回评估根节点的布尔结果。完全二叉树是每个节点有 0 或 2 个子节点的二叉树。叶节点是具有零个子节点的节点。

这道题其实就是考察二叉树的遍历，我们可以使用 DFS 遍历每一个节点，当遇到叶子节点的时候直接返回其布尔值，遇到非叶子结点的时候我们返回其左右子树的 OR 或 AND 运算结果。

时间复杂为 O(N) ，空间复杂度为 O(logN)。

### 解答

	class Solution(object):
	    def evaluateTree(self, root):
	        """
	        :type root: Optional[TreeNode]
	        :rtype: bool
	        """
	        def dfs(node):
	            if not node.left and not node.right:
	                if node.val == 0:
	                    return False
	                return True
	            value = node.val
	            if value == 2:
	                return dfs(node.left) or dfs(node.right)
	            elif value == 3:
	                return dfs(node.left) and dfs(node.right)
	        return dfs(root)

### 运行结果

	
	75 / 75 test cases passed.
	Status: Accepted
	Runtime: 75 ms
	Memory Usage: 14.4 MB

### 原题链接
	
	https://leetcode.com/contest/biweekly-contest-82/problems/evaluate-boolean-binary-tree/


您的支持是我最大的动力
