leetcode  133. Clone Graph（python）




### 描述

Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

	class Node {
	    public int val;
	    public List<Node> neighbors;
	}

Example 1:


![](https://assets.leetcode.com/uploads/2019/11/04/133_clone_graph_question.png)

	Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
	Output: [[2,4],[1,3],[2,4],[1,3]]
	Explanation: There are 4 nodes in the graph.
	1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
	2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
	3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
	4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
	
Example 2:


![](https://assets.leetcode.com/uploads/2020/01/07/graph.png)

	Input: adjList = [[]]
	Output: [[]]
	Explanation: Note that the input contains one empty list. The graph consists of only one node with val = 1 and it does not have any neighbors.

Example 3:

	
	Input: adjList = []
	Output: []
	Explanation: This an empty graph, it does not have any nodes.
	




Note:


	The number of nodes in the graph is in the range [0, 100].
	1 <= Node.val <= 100
	Node.val is unique for each node.
	There are no repeated edges and no self-loops in the graph.
	The Graph is connected and all nodes can be visited starting from the given node.

### 解析

根据题意，给定连接无向图中一个节点的引用。返回图的深度克隆。图中的每个节点都包含一个值 (int) 和一个其邻居的列表 (List[Node])。格式如下：

	class Node {
	    public int val;
	    public List<Node> neighbors;
	}
	
并且每个节点的值与其索引相同，如第一个节点的值为 1 ，第二个节点的值为 2 等等。

其实这道题考察的就是 BFS ，我们就是要根据给定的一个节点引用 node 用 BFS 遍历出所有的节点并以新的值建立节点，在遍历的过程中将它们各自的 neighbors 都填充到新的节点中去。用 node 的值初始化一个新的节点 root ，代码中的 stack 就是用来进行 BFS 的节点遍历，visit 是用来给新的节点填充 neighbors 并且防止重复遍历节点，最后返回 root 即可。如果 node 为空直接返回即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答
				

	class Solution(object):
	    def cloneGraph(self, node):
	        """
	        :type node: Node
	        :rtype: Node
	        """
	        if not node: return node
	        root = Node(node.val)
	        stack = [node]
	        visit = {}
	        visit[node.val] = root
	        while stack:
	            top = stack.pop()
	            for n in top.neighbors:
	                if n.val not in visit:
	                    stack.append(n)
	                    visit[n.val] = Node(n.val)
	                visit[top.val].neighbors.append(visit[n.val])
	        return root
			
### 运行结果

	Runtime: 39 ms, faster than 86.82% of Python online submissions for Clone Graph.
	Memory Usage: 13.8 MB, less than 31.45% of Python online submissions for Clone Graph.


### 原题链接

https://leetcode.com/problems/clone-graph/


您的支持是我最大的动力
