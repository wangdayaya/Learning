leetcode  114. Flatten Binary Tree to Linked List（python）




### 描述



Given the root of a binary tree, flatten the tree into a "linked list":

* The "linked list" should use the same TreeNode class where the right child pointer points to the next node in the list and the left child pointer is always null.
* The "linked list" should be in the same order as a pre-order traversal of the binary tree.


Example 1:

![](https://assets.leetcode.com/uploads/2021/01/14/flaten.jpg)

	Input: root = [1,2,5,3,4,null,6]
	Output: [1,null,2,null,3,null,4,null,5,null,6]

	
Example 2:

	Input: root = []
	Output: []


Example 3:


	Input: root = [0]
	Output: [0]


Note:


	The number of nodes in the tree is in the range [0, 2000].
	-100 <= Node.val <= 100

### 解析

根据题意，给定二叉树的根 root ，将树中的节点压成链表：

* “链表”应该使用相同的 TreeNode 类，其中右子指针指向列表中的下一个节点，左子指针始终为空。
* “链表”的顺序应该与二叉树的前序遍历相同。

解决这道题最简单的方法就是直接使用 DFS 进行前序遍历，把节点都找出来，然后再将它们按照顺序拼接成题目中的“链表”。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def flatten(self, root):
	        """
	        :type root: TreeNode
	        :rtype: None Do not return anything, modify root in-place instead.
	        """
	        nodes = []
	        def dfs(root):
	            if root:
	                nodes.append(root)
	                dfs(root.left)
	                dfs(root.right)
	        dfs(root)
	        N = len(nodes)
	        for i in range(1, N):
	            pre, cur = nodes[i-1], nodes[i]
	            pre.left = None
	            pre.right = cur

### 运行结果

	Runtime: 37 ms, faster than 49.14% of Python online submissions for Flatten Binary Tree to Linked List.
	Memory Usage: 14.2 MB, less than 73.92% of Python online submissions for Flatten Binary Tree to Linked List.

### 解析

题目中给我们提出的要求是使用 O(1) 的空间复杂度来解决这个问题，也就是我们不能使用额外的空间，只能通过指针的变化来操作整个二叉树变成“链表”，通过观察我们发现，其实当某节点的左子树为空，则不需要进行操作，如果某节点的左子树不为空，则左子树中在前序遍历中的最后一个节点是左子树中的最右边的叶子节点，我们只需要将该节点的右子树转移到左子树最右叶子节点之后即可完成“链表”转换。不断重复这个过程即可获得“链表”。

时间复杂度为 O(N) ，空间复杂度为O (1) 。

### 解答
	class Solution(object):
	    def flatten(self, root):
	        """
	        :type root: TreeNode
	        :rtype: None Do not return anything, modify root in-place instead.
	        """
	        while root:
	            if root.left:
	                pre  = nxt = root.left
	                while pre.right:
	                    pre = pre.right
	                pre.right = root.right
	                root.left = None
	                root.right = nxt
	            root = root.right
	        

### 运行结果

	Runtime: 31 ms, faster than 68.59% of Python online submissions for Flatten Binary Tree to Linked List.
	Memory Usage: 14.3 MB, less than 73.92% of Python online submissions for Flatten Binary Tree to Linked List.

### 原题链接

https://leetcode.com/problems/flatten-binary-tree-to-linked-list/


您的支持是我最大的动力
