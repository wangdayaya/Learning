leetcode  429. N-ary Tree Level Order Traversal（python）




### 描述


Given an n-ary tree, return the level order traversal of its nodes' values.

Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See examples).


Example 1:

![](https://assets.leetcode.com/uploads/2018/10/12/narytreeexample.png)

	Input: root = [1,null,3,2,4,null,5,6]
	Output: [[1],[3,2,4],[5,6]]

	
Example 2:

![](https://assets.leetcode.com/uploads/2019/11/08/sample_4_964.png)

	Input: root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
	Output: [[1],[2,3,4,5],[6,7,8,9,10],[11,12,13],[14]]




Note:

	The height of the n-ary tree is less than or equal to 1000
	The total number of nodes is between [0, 10^4]


### 解析

根据题意，给定一个 n 叉树，按层级遍历顺序返回其节点值。题目中还给出了层级遍历的概念，以及输入的 N 叉树形式。

这道题其实就是考察使用层级遍历方式完成 N 叉树的层级遍历，大体思路和二叉树的层级遍历一样，我们使用一个队列 deque 来保存节点，然后使用循环进行遍历 deque 的操作，每次循环我们将当前层的节点从左到右依次弹出并放入 tmp 中进行记录，并将每个弹出节点的下一层的节点挨个依次追加到 deque 的末尾，当前层遍历结束之后将 tmp 追加到 result 中，如此循环结束之后，最后返回 result 就是按照层级遍历的结果。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def levelOrder(self, root):
	        """
	        :type root: Node
	        :rtype: List[List[int]]
	        """
	        if not root:
	            return []
	        deque = collections.deque([root])
	        result = []
	        while deque:
	            tmp = []
	            for _ in range(len(deque)):
	                x = deque.popleft()
	                tmp.append(x.val)
	                for node in x.children:
	                    deque.append(node)
	            result.append(tmp)
	        return result

### 运行结果

	Runtime: 60 ms, faster than 52.05% of Python online submissions for N-ary Tree Level Order Traversal.
	Memory Usage: 16.6 MB, less than 19.56% of Python online submissions for N-ary Tree Level Order Traversal.

### 原题链接

	https://leetcode.com/problems/n-ary-tree-level-order-traversal/



您的支持是我最大的动力
