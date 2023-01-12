leetcode 2385. Amount of Time for Binary Tree to Be Infected （python）




### 描述

You are given the root of a binary tree with unique values, and an integer start. At minute 0, an infection starts from the node with value start. Each minute, a node becomes infected if:

* The node is currently uninfected.
* The node is adjacent to an infected node.

Return the number of minutes needed for the entire tree to be infected.



Example 1:

![](https://assets.leetcode.com/uploads/2022/06/25/image-20220625231744-1.png)

	Input: root = [1,5,3,null,4,10,6,9,2], start = 3
	Output: 4
	Explanation: The following nodes are infected during:
	- Minute 0: Node 3
	- Minute 1: Nodes 1, 10 and 6
	- Minute 2: Node 5
	- Minute 3: Node 4
	- Minute 4: Nodes 9 and 2
	It takes 4 minutes for the whole tree to be infected so we return 4.

	
Example 2:

![](https://assets.leetcode.com/uploads/2022/06/25/image-20220625231812-2.png)
	
	Input: root = [1], start = 1
	Output: 0
	Explanation: At minute 0, the only node in the tree is infected so we return 0.



Note:

	The number of nodes in the tree is in the range [1, 10^5].
	1 <= Node.val <= 10^5
	Each node has a unique value.
	A node with a value of start exists in the tree.


### 解析

根据题意，给定具有唯一值的二叉树的根 root 和一个整数 start 。 在第 0 分钟，感染从值为 start 的节点开始。 每分钟，如果出现以下情况，节点就会被感染：

* 该节点当前未受感染。
* 该节点与受感染的节点相邻。

返回整个树被感染所需的分钟数。

这个题很明显整体需要 BFS 的方式进行解题，但是关键在于这是一颗树，我们要使用两个字典 parent 和 child 来记录每个节点的父节点和子节点有哪些，这个过程需要最基础的 DFS 进行解决，然后我们得到 parent 和 child 之后，就可以按照 BFS 的思路从 start 开始模拟病毒扩散的效果来解题，使用 result 来记录扩散时间即可。虽然代码量上比较大，但是很好理解，上半部分就是 DFS 过程，下半部分就是 BFS 过程。当然代码还可以优化，有兴趣的同学可以尝试一下。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def __init__(self):
	        self.parent = {}
	        self.child = collections.defaultdict(list)
	    def amountOfTime(self, root, start):
	        def dfs(root):
	            if not root:
	                return
	            if root.left:
	                self.parent[root.left.val] = root
	                self.child[root.val].append(root.left)
	                dfs(root.left)
	            if root.right:
	                self.parent[root.right.val] = root
	                self.child[root.val].append(root.right)
	                dfs(root.right)
	        dfs(root)
	        result = 0
	        if start == root.val and not self.parent and not self.child:
	            return result
	        stack = [start]
	        visited = set()
	        while stack:
	            for _ in range(len(stack)):
	                node = stack.pop(0)
	                visited.add(node)
	                if node in self.parent and self.parent[node].val not in visited:
	                    stack.append(self.parent[node].val)
	                for c in self.child[node]:
	                    if c.val not in visited:
	                        stack.append(c.val)
	            if stack:
	                result += 1
	        return result


### 运行结果

	
	80 / 80 test cases passed.
	Status: Accepted
	Runtime: 2872 ms
	Memory Usage: 176.1 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-307/problems/amount-of-time-for-binary-tree-to-be-infected/


您的支持是我最大的动力
