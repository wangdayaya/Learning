leetcode  1379. Find a Corresponding Node of a Binary Tree in a Clone of That Tree（python）

### 描述


Given two binary trees original and cloned and given a reference to a node target in the original tree.

The cloned tree is a copy of the original tree.

Return a reference to the same node in the cloned tree.

Note that you are not allowed to change any of the two trees or the target node and the answer must be a reference to a node in the cloned tree.

Follow up: Solve the problem if repeated values on the tree are allowed.


Example 1:

![](https://assets.leetcode.com/uploads/2020/02/21/e1.png)

	Input: tree = [7,4,3,null,null,6,19], target = 3
	Output: 3
	Explanation: In all examples the original and cloned trees are shown. The target node is a green node from the original tree. The answer is the yellow node from the cloned tree.


	
Example 2:
![](https://assets.leetcode.com/uploads/2020/02/21/e2.png)
	
	Input: tree = [7], target =  7
	Output: 7



Example 3:


![](https://assets.leetcode.com/uploads/2020/02/21/e3.png)

	Input: tree = [8,null,6,null,5,null,4,null,3,null,2,null,1], target = 4
	Output: 4
	
Example 4:

![](https://assets.leetcode.com/uploads/2020/02/21/e4.png)

	Input: tree = [1,2,3,4,5,6,7,8,9,10], target = 5
	Output: 5

	
Example 5:

![](https://assets.leetcode.com/uploads/2020/02/21/e5.png)

	Input: tree = [1,2,null,3], target = 2
	Output: 2


Note:

	The number of nodes in the tree is in the range [1, 10^4].
	The values of the nodes of the tree are unique.
	target node is a node from the original tree and is not null.


### 解析


根据题意，给出了一个原二叉树 original ，又给出了一个复制的二叉树 cloned ，和 original 一模一样，给出了一个在 original 上的目标节点 target ，让我们得到在 cloned 上同样的节点的引用，思路比较简单，其实都没有用到 original ，使用了递归的思想：

当 cloned 根结点的值和 target 节点的值一样的时候，返回 clone ，否则使用递归遍历当前节点的左右两个子树。

### 解答
				

	class TreeNode(object):
	    def __init__(self, x):
	        self.val = x
	        self.left = None
	        self.right = None
	
	class Solution(object):
	    def getTargetCopy(self, original, cloned, target):
	        """
	        :type original: TreeNode
	        :type cloned: TreeNode
	        :type target: TreeNode
	        :rtype: TreeNode
	        """
	        if not cloned: return
	        if cloned.val==target.val:
	            return cloned
	        return self.getTargetCopy(original, cloned.left, target) or self.getTargetCopy(original, cloned.right, target)
	           
			
### 运行结果

	Runtime: 740 ms, faster than 72.58% of Python online submissions for Find a Corresponding Node of a Binary Tree in a Clone of That Tree.
	Memory Usage: 30.2 MB, less than 91.94% of Python online submissions for Find a Corresponding Node of a Binary Tree in a Clone of That Tree.

### 解析


直接使用栈 stack ，保存所有的节点，然后判断当前节点的值是否和 target 相同，如果相同返回当前节点的引用。

### 解答
	class TreeNode(object):
	    def __init__(self, x):
	        self.val = x
	        self.left = None
	        self.right = None
	
	class Solution(object):
	    def getTargetCopy(self, original, cloned, target):
	        """
	        :type original: TreeNode
	        :type cloned: TreeNode
	        :type target: TreeNode
	        :rtype: TreeNode
	        """
	        stack = [cloned]
	        while stack:
	            curt = stack.pop(0)
	            if curt and curt.val == target.val:
	                return curt
	            if curt.left:
	                stack.append(curt.left)
	            if curt.right:
	                stack.append(curt.right)
### 运行结果

	Runtime: 736 ms, faster than 78.23% of Python online submissions for Find a Corresponding Node of a Binary Tree in a Clone of That Tree.
	Memory Usage: 30.5 MB, less than 33.87% of Python online submissions for Find a Corresponding Node of a Binary Tree in a Clone of That Tree.
原题链接：https://leetcode.com/problems/find-a-corresponding-node-of-a-binary-tree-in-a-clone-of-that-tree/



您的支持是我最大的动力
