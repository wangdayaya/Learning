leetcode  2049. Count Nodes With the Highest Score（python）

### 描述

There is a binary tree rooted at 0 consisting of n nodes. The nodes are labeled from 0 to n - 1. You are given a 0-indexed integer array parents representing the tree, where parents[i] is the parent of node i. Since node 0 is the root, parents[0] == -1.

Each node has a score. To find the score of a node, consider if the node and the edges connected to it were removed. The tree would become one or more non-empty subtrees. The size of a subtree is the number of the nodes in it. The score of the node is the product of the sizes of all those subtrees.

Return the number of nodes that have the highest score.





Example 1:

![](https://assets.leetcode.com/uploads/2021/10/03/example-1.png)

	Input: parents = [-1,2,0,2,0]
	Output: 3
	Explanation:
	- The score of node 0 is: 3 * 1 = 3
	- The score of node 1 is: 4 = 4
	- The score of node 2 is: 1 * 1 * 2 = 2
	- The score of node 3 is: 4 = 4
	- The score of node 4 is: 4 = 4
	The highest score is 4, and three nodes (node 1, node 3, and node 4) have the highest score.

	
Example 2:

![](https://assets.leetcode.com/uploads/2021/10/03/example-2.png)

	Input: parents = [-1,2,0]
	Output: 2
	Explanation:
	- The score of node 0 is: 2 = 2
	- The score of node 1 is: 2 = 2
	- The score of node 2 is: 1 * 1 = 1
	The highest score is 2, and two nodes (node 0 and node 1) have the highest score.





Note:

	n == parents.length
	2 <= n <= 10^5
	parents[0] == -1
	0 <= parents[i] <= n - 1 for i != 0
	parents represents a valid binary tree.


### 解析

根据题意，存在一个以 0 为根的二叉树，由 n 个节点组成。 节点被标记为从 0 到 n - 1 。给你一个 0 索引的整数数组 parent，表示树，其中 parents[i] 是节点 i 的父节点。 由于节点 0 是根节点，因此 parents[0] == -1 。每个节点都有一个分数，要查找节点的分数，请考虑是否删除了该节点及其连接的边。 该树将成为一棵或多棵非空子树。 子树的大小是其中的节点数。 节点的分数是所有这些子树的大小的乘积。返回得分最大的节点个数。


因为去掉一个节点之后，模式上会有生成三部分，节点上面的部分的节点个数 p ，以及节点的左子树个数 l 、节点的右子树个数 r，将 p\*l\*r 就是当前节点的分数。所以这道题其实本质上就是求子树节点个数，我们定义一个 DFS 函数，返回以当前节点为根结点时，其子树包含的节点个数，在递归函数的中间，我们顺便就可以得到左子树节点个数、右子树节点个数、节点上面的节点个数，只要保证他们都大于 0 ，互相乘起来就能得到当前节点的分数，最后找出最大分数的个数即可。

### 解答
				

	class Solution(object):
	    def __init__(self):
	        self.children = None
	        self.result = {}
	        
	    def countHighestScoreNodes(self, parents):
	        """
	        :type parents: List[int]
	        :rtype: int
	        """
	        N = len(parents)
	        self.children = [[] for _ in range(N)]
	        for i in range(N):
	            if parents[i]!=-1:
	                self.children[parents[i]].append(i)
	        self.dfs(0);
	        return self.result[max(self.result)]
	        
	    def dfs(self, root):
	        count = 0
	        L = []
	        for node in self.children[root]:
	            L.append(self.dfs(node))
	            count += L[-1]
	        score = 1
	        if len(self.children)-1-count>0:
	            score *= len(self.children)-1-count
	        for s in L:
	            if s>0:
	                score *= s
	        if score not in self.result:
	            self.result[score] = 1
	        else:
	            self.result[score] += 1
	        return count+1
	        
            	      
			
### 运行结果

	Runtime: 2016 ms, faster than 27.12% of Python online submissions for Count Nodes With the Highest Score.
	Memory Usage: 105.9 MB, less than 42.37% of Python online submissions for Count Nodes With the Highest Score.


原题链接：https://leetcode.com/problems/count-nodes-with-the-highest-score/



您的支持是我最大的动力
