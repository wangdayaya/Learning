leetcode 1008. Construct Binary Search Tree from Preorder Traversal （python）

### 描述


Given an array of integers preorder, which represents the preorder traversal of a BST (i.e., binary search tree), construct the tree and return its root.

It is guaranteed that there is always possible to find a binary search tree with the given requirements for the given test cases.

A binary search tree is a binary tree where for every node, any descendant of Node.left has a value strictly less than Node.val, and any descendant of Node.right has a value strictly greater than Node.val.

A preorder traversal of a binary tree displays the value of the node first, then traverses Node.left, then traverses Node.right.

 


Example 1:

![](https://assets.leetcode.com/uploads/2019/03/06/1266.png)

	Input: preorder = [8,5,1,7,10,12]
	Output: [8,5,10,1,7,null,12]

	
Example 2:

	Input: preorder = [1,3]
	Output: [1,null,3]




Note:

	1 <= preorder.length <= 100
	1 <= preorder[i] <= 10^8
	All the values of preorder are unique.


### 解析

根据题意，就是给出了一个二叉树的前序遍历结果，然后要求根据前序遍历构造一棵树，并且返回树的根节点。至于二叉树的定义和前序遍历的定义，题目中已经解释了，这里不再赘述。

当时自己解题的时候思路很乱，一直没有细扣例子中的规律，所以一直头大，直到看到了高人的解法豁然开朗。仔细研究发现其实就是个找规律的题目。


前序遍历有一个明显的特点，第一个出现的元素肯定是属于根节点，从第二个元素开始后面的元素肯定是属于左右子树的，而左子树都比根节点小，右子树都比右子树大，前序遍历会先遍历完左子树再遍历右子树，所以从第二个元素开始往后找出大于根节点的元素索引 i ，这就是右子树的根节点 ，左子树的元素则在 preorder[1:i] 列表中，右子树的元素则在 preorder[i:] 列表中，递归执行这个过程得到的树即为结果。


### 解答
				

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def bstFromPreorder(self, preorder):
	        """
	        :type preorder: List[int]
	        :rtype: TreeNode
	        """
	        if not preorder: return None
	        root = TreeNode(preorder[0])
	        N = len(preorder)
	        i = 1
	        while i < N:
	            if preorder[i] > root.val:
	                break
	            i += 1
	        root.left = self.bstFromPreorder(preorder[1:i])
	        root.right = self.bstFromPreorder(preorder[i:])
	        return root
            	      
			
### 运行结果

	Runtime: 25 ms, faster than 43.13% of Python online submissions for Construct Binary Search Tree from Preorder Traversal.
	Memory Usage: 13.6 MB, less than 21.80% of Python online submissions for Construct Binary Search Tree from Preorder Traversal.

### 解析

遍历列表元素也可以实现，原理和上面类似。

* 首先用 preorder[0] 初始化 root 根结点，使用 stack 来找不同子树的根节点
* 从第二个元素开始遍历 preorder ，元素有两种情况：
（1）当元素小于 stack[-1] 时候，那当前元素属于的 stack[-1]  左子树，并将当前的节点加入到 stack 中
（2）否则，当元素大于 stack[-1] ，此时不断 stack.pop() 找到适合它的根节点，找到的根节点的右子树就是当前元素 ，并将当前结点加入到 stack 中
* 遍历结束得到的 root 就是答案

### 解答

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def bstFromPreorder(self, preorder):
	        """
	        :type preorder: List[int]
	        :rtype: TreeNode
	        """
	        root = TreeNode(preorder[0])
	        stack = [root]
	        for value in preorder[1:]:
	            if value < stack[-1].val:
	                left = TreeNode(value)
	                stack[-1].left = left
	                stack.append(left)
	            else:
	                while stack and stack[-1].val<value:
	                    tmp = stack.pop()
	                right = TreeNode(value)
	                tmp.right = right
	                stack.append(right)
	        return root

### 运行结果

	Runtime: 39 ms, faster than 5.69% of Python online submissions for Construct Binary Search Tree from Preorder Traversal.
	Memory Usage: 13.7 MB, less than 21.80% of Python online submissions for Construct Binary Search Tree from Preorder Traversal.

原题链接：https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal/



您的支持是我最大的动力
