leetcode 701. Insert into a Binary Search Tree （python）

### 描述



You are given the root node of a binary search tree (BST) and a value to insert into the tree. Return the root node of the BST after the insertion. It is guaranteed that the new value does not exist in the original BST.

Notice that there may exist multiple valid ways for the insertion, as long as the tree remains a BST after insertion. You can return any of them.

Example 1:

![](https://assets.leetcode.com/uploads/2020/10/05/insertbst.jpg)

	Input: root = [4,2,7,1,3], val = 5
	Output: [4,2,7,1,3,5]
	Explanation: Another accepted tree is:

![](https://assets.leetcode.com/uploads/2020/10/05/bst.jpg)

Example 2:

	
	Input: root = [40,20,60,10,30,50,70], val = 25
	Output: [40,20,60,10,30,50,70,null,null,25]

Example 3:


	Input: root = [4,2,7,1,3,null,null,null,null,null,null], val = 5
	Output: [4,2,7,1,3,5]
	


Note:
	
	The number of nodes in the tree will be in the range [0, 10^4].
	-10^8 <= Node.val <= 10^8
	All the values Node.val are unique.
	-10^8 <= val <= 10^8
	It's guaranteed that val does not exist in the original BST.


### 解析


根据题意，就是给出了一个二叉树的根节点 root ，然后给出了一个值 val ，题目要求我们将 val 值的节点插入到二叉树中，并且返回根节点，题目中保证 val 是不存在于 root 的值，而且可能存在多种二叉树，只需要返回其中任意一个就可以了。

还是老办法使用递归进行解题，这里只需要知道二叉树的基本性质，那就是左子树的值比根节点的值小，右子树的值比根节点的值大。然后使用递归：

* 如果根节点 root 为空，则使用 val 初始化一个节点并返回
* 如果根节点 root 的值小于 val ，说明 val 应该在 root 的右子树，所以递归对 root 的右子树进行递归节点插入
* 如果根节点 root 的值大于 val ，说明 val 应该在 root 的左子树，所以递归对 root 的左子树进行递归节点插入
* 最后返回当前根节点 root 

### 解答
				
	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def insertIntoBST(self, root, val):
	        """
	        :type root: TreeNode
	        :type val: int
	        :rtype: TreeNode
	        """
	        if not root:
	            node = TreeNode(val)
	            return node
	        if root.val < val:
	            root.right = self.insertIntoBST(root.right, val)
	        else:
	            root.left = self.insertIntoBST(root.left, val)
	        return root

            	      
			
### 运行结果

	Runtime: 166 ms, faster than 23.65% of Python online submissions for Insert into a Binary Search Tree.
	Memory Usage: 17.6 MB, less than 41.28% of Python online submissions for Insert into a Binary Search Tree.


### 解析

其实用比较笨拙的思路，可以分几步做：

* 初始化一个 L 空列表用来存放所有的节点值
* 定义 getListFromBST 函数使用递归遍历所有的节点的值，都存放入 L 中
* 将 val 也加入 L 中然后对 L 进行排序
* 定义函数 sortedListToBST 使用递归对已经升序的列表构建成二叉树，这个操作和这道题一摸一样 [leetcode 108. Convert Sorted Array to Binary Search Tree（python）](https://juejin.cn/post/7021322404328636452) 

含明显这种方法繁琐了很多，不如上一种方法简洁明了。
### 解答

	class TreeNode(object):
	    def __init__(self, val=0, left=None, right=None):
	        self.val = val
	        self.left = left
	        self.right = right
	class Solution(object):
	    def __init__(self):
	        self.L = []
	        
	    def getListFromBST(self, root):
	        if not root:return 
	        self.getListFromBST(root.left)
	        self.L.append(root.val)
	        self.getListFromBST(root.right)
	        
	    def sortedListToBST(self, nums):
	        if not nums: return 
	        MID = len(nums) // 2
	        root = TreeNode(nums[MID])
	        root.left = self.sortedListToBST(nums[:MID])
	        root.right = self.sortedListToBST(nums[MID+1:])
	        return root
	        
	    def insertIntoBST(self, root, val):
	        """
	        :type root: TreeNode
	        :type val: int
	        :rtype: TreeNode
	        """
	        self.getListFromBST(root)
	        self.L.append(val)
	        self.L.sort()
	        return self.sortedListToBST(self.L)

### 运行结果

	Runtime: 389 ms, faster than 5.01% of Python online submissions for Insert into a Binary Search Tree.
	Memory Usage: 19.4 MB, less than 16.83% of Python online submissions for Insert into a Binary Search Tree.


原题链接：https://leetcode.com/problems/insert-into-a-binary-search-tree/



您的支持是我最大的动力
