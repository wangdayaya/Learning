leetcode  1261. Find Elements in a Contaminated Binary Tree（python）

### 描述

Given a binary tree with the following rules:

* root.val == 0
* If treeNode.val == x and treeNode.left != null, then treeNode.left.val == 2 * x + 1
* If treeNode.val == x and treeNode.right != null, then treeNode.right.val == 2 * x + 2

Now the binary tree is contaminated, which means all treeNode.val have been changed to -1.

Implement the FindElements class:

* FindElements(TreeNode* root) Initializes the object with a contaminated binary tree and recovers it.
* bool find(int target) Returns true if the target value exists in the recovered binary tree.



Example 1:
![](https://assets.leetcode.com/uploads/2019/11/06/untitled-diagram-4-1.jpg)

	Input
	["FindElements","find","find"]
	[[[-1,null,-1]],[1],[2]]
	Output
	[null,false,true]
	Explanation
	FindElements findElements = new FindElements([-1,null,-1]); 
	findElements.find(1); // return False 
	findElements.find(2); // return True 


	
Example 2:

![](https://assets.leetcode.com/uploads/2019/11/06/untitled-diagram-4.jpg)

	Input
	["FindElements","find","find","find"]
	[[[-1,-1,-1,-1,-1]],[1],[3],[5]]
	Output
	[null,true,true,false]
	Explanation
	FindElements findElements = new FindElements([-1,-1,-1,-1,-1]);
	findElements.find(1); // return True
	findElements.find(3); // return True
	findElements.find(5); // return False

Example 3:


![](https://assets.leetcode.com/uploads/2019/11/07/untitled-diagram-4-1-1.jpg)

	Input
	["FindElements","find","find","find","find"]
	[[[-1,null,-1,-1,null,-1]],[2],[3],[4],[5]]
	Output
	[null,true,false,false,true]
	Explanation
	FindElements findElements = new FindElements([-1,null,-1,-1,null,-1]);
	findElements.find(2); // return True
	findElements.find(3); // return False
	findElements.find(4); // return False
	findElements.find(5); // return True
	



Note:
	
	TreeNode.val == -1
	The height of the binary tree is less than or equal to 20
	The total number of nodes is between [1, 10^4]
	Total calls of find() is between [1, 10^4]
	0 <= target <= 10^6


### 解析

根据题意，给出了一棵树，但是其中只能看到树的结构，因为树被污染所有节点的值都是 -1 ，但是节点的值可以复现恢复，那就是根节点为 0 ，左节点的值是其父节点的值的两倍加一，右节点的值是其父节点的值的两倍加二。题目要求我们使用 \_\_init__ 函数先恢复这颗树，然后使用 find 函数判断 target 是否存在于树中。

思路比较简单，就是用递归直接将树的节点的值都计算出来存在一个列表当中，然后判断 target 是否在列表中即可。其实这道题看起来复杂，实际上蛮简单的。


### 解答
				
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class FindElements(object):
	    def __init__(self, root):
	        """
	        :type root: TreeNode
	        """
	        self.vals = []
	        def dfs(root, val):
	            if not root:
	                return
	            self.vals.append(val)
	            if root.left:
	                dfs(root.left, val*2+1)
	            if root.right:
	                dfs(root.right, val*2+2)
	        dfs(root, 0)
	
	
	    def find(self, target):
	        """
	        :type target: int
	        :rtype: bool
	        """
	        if target in self.vals:
	            return True
	        return False

            	      
			
### 运行结果

	Runtime: 622 ms, faster than 6.67% of Python online submissions for Find Elements in a Contaminated Binary Tree.
	Memory Usage: 19.2 MB, less than 76.67% of Python online submissions for Find Elements in a Contaminated Binary Tree.


### 解析

也可以使用队列来解答这个题，构建树的过程和上面过程类似，不再赘述。


### 解答

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class FindElements(object):
	    def __init__(self, root):
	        self.A = set()
	        queue = collections.deque([[root,0]])
	        while queue:
	            n,x = queue.popleft()
	            self.A.add(x)
	            if n.left:
	                queue.append( [n.left  , 2*x+1] )
	            if n.right:
	                queue.append( [n.right , 2*x+2] )
	                
	    def find(self, target):
	        return target in self.A
	        
### 运行结果

	Runtime: 143 ms, faster than 33.33% of Python online submissions for Find Elements in a Contaminated Binary Tree.
	Memory Usage: 19.7 MB, less than 13.33% of Python online submissions for Find Elements in a Contaminated Binary Tree.    

原题链接：https://leetcode.com/problems/find-elements-in-a-contaminated-binary-tree/



您的支持是我最大的动力
