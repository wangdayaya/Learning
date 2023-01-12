leetcode  2096. Step-By-Step Directions From a Binary Tree Node to Another（python）

### 描述


You are given the root of a binary tree with n nodes. Each node is uniquely assigned a value from 1 to n. You are also given an integer startValue representing the value of the start node s, and a different integer destValue representing the value of the destination node t.

Find the shortest path starting from node s and ending at node t. Generate step-by-step directions of such path as a string consisting of only the uppercase letters 'L', 'R', and 'U'. Each letter indicates a specific direction:

* 'L' means to go from a node to its left child node.
* 'R' means to go from a node to its right child node.
* 'U' means to go from a node to its parent node.

Return the step-by-step directions of the shortest path from node s to node t.




Example 1:

![](https://assets.leetcode.com/uploads/2021/11/15/eg1.png)

	Input: root = [5,1,2,3,null,6,4], startValue = 3, destValue = 6
	Output: "UURL"
	Explanation: The shortest path is: 3 → 1 → 5 → 2 → 6.

	
Example 2:

![](https://assets.leetcode.com/uploads/2021/11/15/eg2.png)

	Input: root = [2,1], startValue = 2, destValue = 1
	Output: "L"
	Explanation: The shortest path is: 2 → 1.




Note:

	The number of nodes in the tree is n.
	2 <= n <= 10^5
	1 <= Node.val <= n
	All the values in the tree are unique.
	1 <= startValue, destValue <= n
	startValue != destValue


### 解析

根据题意，给定具有 n 个节点的二叉树的根 root 。 每个节点都被唯一分配了一个从 1 到 n 的值。 还给了一个表示起始节点 s 值的整数 startValue，以及一个表示目标节点 t 值的不同整数 destValue。找出从节点 s 开始到节点 t 结束的最短路径。 将此类路径的逐步方向生成为仅包含大写字母 “L” 、“R” 和 “U” 的字符串。 每个字母表示一个特定的方向：

* 'L' 表示从一个节点到它的左子节点。
* 'R' 表示从一个节点到它的右子节点。
* 'U' 意味着从一个节点到它的父节点。

返回从节点 s 到节点 t 的最短路径的逐步方向。

其实这道题的本质就是考察最低公共祖先，我们可以用最暴力的方法，将根节点到 startValue 的路径找到，再将根节点到 destValue 的路径找到，然后从两条路径中找出最低公共祖先节点，以此为中转点，将 startValue 到 destValue 的路径用方向字符串拼接起来即可，但是这种方法容易超时。
### 解答
				

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	        
	class Solution(object):
	    def getDirections(self, root, startValue, destValue):
	        """
	        :type root: Optional[TreeNode]
	        :type startValue: int
	        :type destValue: int
	        :rtype: str
	        """
	        paths  = []
	        result = ''
	        def dfs(root, path, t):
	            if not root: return 
	            if root.val == t:
	                paths.append(path+[root])
	                return 
	            if root.left:
	                dfs(root.left, path+[root], t)
	            if root.right:
	                dfs(root.right, path+[root], t)
	        dfs(root, [], startValue)
	        dfs(root, [], destValue)
	        L = paths[0]
	        R = paths[1]
	        idx = 0
	        while idx<min(len(L),len(R)):
	            if L[idx].val!=R[idx].val:
	                break
	            idx += 1
	            
	        for i in range(len(L)-idx):
	            result += 'U'
	        
	        for i in range(idx,len(R)):
	            if R[i-1].left and R[i].val == R[i-1].left.val:
	                result += 'L'
	            elif R[i-1].right and R[i].val == R[i-1].right.val:
	                result += 'R'
	                
	        return result
	            
	        
	        
	                
	                
            	      
			
### 运行结果


	Time Limit Exceeded
	
	
### 解析

其实上面的解法整体思路是对的，只是有很多步骤浪费了时间，我们可以在遍历节点找 startValue 和 destValue 节点的时候，我们就把方向字符都记录下来，然后再找最低公共最先节点，将 LCA 左半部分的方向字符都变成 U ，最后将 LCA 左右两个方向的字符都拼接起来就是答案。
	              

### 解答

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	        
	class Solution(object):
	    def getDirections(self, root, startValue, destValue):
	        """
	        :type root: Optional[TreeNode]
	        :type startValue: int
	        :type destValue: int
	        :rtype: str
	        """
	        L = []
	        R = []
	        P_L = []
	        P_R = []
	        self.dfs(root, L, P_L, startValue)
	        self.dfs(root, R, P_R, destValue)
	        idx = 0
	        while idx<min(len(L), len(R)) and L[idx] == R[idx]:
	            idx += 1
	        for i in range(idx, len(P_L)): P_L[i]= 'U'
	        return ''.join(P_L[idx:]) + ''.join(P_R[idx:])
	        
	       
	    def dfs(self, node, values, path, t):
	        if not node: return False
	        if node.val == t: return True
	        if node.left:
	            values.append(node.left.val)
	            path.append('L')
	            if self.dfs(node.left, values, path, t):
	                return True
	            values.pop()
	            path.pop()
	        if node.right:
	            values.append(node.right.val)
	            path.append('R')
	            if self.dfs(node.right, values, path, t):
	                return True
	            values.pop()
	            path.pop()
	        return False
	            
	            

### 运行结果


	Runtime: 972 ms, faster than 88.33% of Python online submissions for Step-By-Step Directions From a Binary Tree Node to Another.
	Memory Usage: 136.1 MB, less than 71.67% of Python online submissions for Step-By-Step Directions From a Binary Tree Node to Another.

原题链接：https://leetcode.com/problems/step-by-step-directions-from-a-binary-tree-node-to-another/



您的支持是我最大的动力
