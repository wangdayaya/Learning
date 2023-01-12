leetcode  173. Binary Search Tree Iterator（python）




### 描述

Implement the BSTIterator class that represents an iterator over the in-order traversal of a binary search tree (BST):

* BSTIterator(TreeNode root) Initializes an object of the BSTIterator class. The root of the BST is given as part of the constructor. The pointer should be initialized to a non-existent number smaller than any element in the BST.
* boolean hasNext() Returns true if there exists a number in the traversal to the right of the pointer, otherwise returns false.
* int next() Moves the pointer to the right, then returns the number at the pointer.

Notice that by initializing the pointer to a non-existent smallest number, the first call to next() will return the smallest element in the BST.

You may assume that next() calls will always be valid. That is, there will be at least a next number in the in-order traversal when next() is called.



Example 1:


![](https://assets.leetcode.com/uploads/2018/12/25/bst-tree.png)

	Input
	["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
	[[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
	Output
	[null, 3, 7, true, 9, true, 15, true, 20, false]
	
	Explanation
	BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]);
	bSTIterator.next();    // return 3
	bSTIterator.next();    // return 7
	bSTIterator.hasNext(); // return True
	bSTIterator.next();    // return 9
	bSTIterator.hasNext(); // return True
	bSTIterator.next();    // return 15
	bSTIterator.hasNext(); // return True
	bSTIterator.next();    // return 20
	bSTIterator.hasNext(); // return False
	



Note:


	The number of nodes in the tree is in the range [1, 10^5].
	0 <= Node.val <= 10^6
	At most 10^5 calls will be made to hasNext, and next.

### 解析

根据题意，实现 BSTIterator 类，该类表示在二叉搜索树 (BST) 的中序遍历上的迭代器：

* BSTIterator(TreeNode root) 初始化 BSTIterator 类的对象。 BST 的根作为构造函数的一部分给出。 应初始化为一个不存在的数字，该数字小于 BST 中的任何元素。
* boolean hasNext() 如果指针右侧的遍历中存在数字，则返回 true，否则返回 false。
* int next() 将指针向右移动，然后返回指针处的数字。

通过将指针初始化为不存在的最小数字，第一次调用 next() 将返回 BST 中的最小元素。我们可以假设 next() 调用将始终有效。 也就是说，在调用 next() 时，有序遍历中至少会有一个数字。

其实这道题就是考察的二叉树的中序遍历，我们在初始化的时候把所有的值按照中序遍历的顺序存入一个列表中，然后在执行 next() 函数和 hasNext() 的时候只需要对列表进行弹出元素和判断长度的操作即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答
				
	class BSTIterator(object):
	    def __init__(self, root):
	        """
	        :type root: TreeNode
	        """
	        self.L = []
	        self.inorder(root)
	    
	    def inorder(self, root):
	        if root:
	            self.inorder(root.left)
	            self.L.append(root.val)
	            self.inorder(root.right)
	
	    def next(self):
	        """
	        :rtype: int
	        """
	        return self.L.pop(0)
	        
	
	    def hasNext(self):
	        """
	        :rtype: bool
	        """
	        return len(self.L)!=0

            	      
			
### 运行结果

	Runtime: 122 ms, faster than 12.85% of Python online submissions for Binary Search Tree Iterator.
	Memory Usage: 21.8 MB, less than 85.59% of Python online submissions for Binary Search Tree Iterator.


### 原题链接



https://leetcode.com/problems/binary-search-tree-iterator/


您的支持是我最大的动力
