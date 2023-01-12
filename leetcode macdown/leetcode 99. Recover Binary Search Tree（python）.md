leetcode 99. Recover Binary Search Tree （python）




### 描述


You are given the root of a binary search tree (BST), where the values of exactly two nodes of the tree were swapped by mistake. Recover the tree without changing its structure.


Example 1:


![](https://assets.leetcode.com/uploads/2020/10/28/recover1.jpg)

	Input: root = [1,3,null,null,2]
	Output: [3,1,null,null,2]
	Explanation: 3 cannot be a left child of 1 because 3 > 1. Swapping 1 and 3 makes the BST valid.
	
Example 2:


![](https://assets.leetcode.com/uploads/2020/10/28/recover2.jpg)

	Input: root = [3,1,4,null,null,2]
	Output: [2,1,4,null,null,3]
	Explanation: 2 cannot be in the right subtree of 3 because 2 < 3. Swapping 2 and 3 makes the BST valid.



Note:

	The number of nodes in the tree is in the range [2, 1000].
	-2^31 <= Node.val <= 2^31 - 1


### 解析


根据题意，给定二叉搜索树 (BST) 的 root ，其中树恰好两个节点的值被错误地交换了。 在不改变其结构的情况下恢复树正确的样子。

 其实这道题考察的就是二叉树的中序遍历，因为树种正好有一对节点错误地交换了值，如例子一中，我们发现使用中序遍历的正常结果应该为 ：1-2-3 ，但是此时却成了 3-2-1 ，我会发现中序遍历的正常规律是升序状态的，所以我们只要找到某个值比前一个值小，说明违反了升序规律，那么肯定是发生错误的节点。
 
 这里有难度的是有两种情况，如某个树的正确中序为：3-5-8-10-13 ，可以出现 5-3-8-10-13 的错误序列（两个错误的位置刚好相邻），可能出现 3-13-8-10-5 的错误序列（两个错误的位置不相邻），所以我们使用常规的解法将所有节点的值按照中序遍历放入列表中，找错误的对即可，但是大可不必，我们使用 O(1) 的空间来解决，只要定义一个 pre 保存前一个节点，如果 pre.val > root.val 就是错误的地方，找到第一个错误点，再找第二个错误的点即可，最后将这两个节点的值交换即可。
 
 时间复杂度为 O(N) ，空间复杂度为 O(H) ，H 为递归栈的深度。

### 解答
				

	class Solution(object):
	    def __init__(self):
	        self.pre = TreeNode(float('-inf'))
	        self.first = self.second = None
	
	    def recoverTree(self, root):
	        self.inorder(root)
	        self.first.val, self.second.val = self.second.val, self.first.val
	
	    def inorder(self, root):
	        if not root: return
	        self.inorder(root.left)
	        if not self.first and self.pre.val > root.val:
	            self.first = self.pre
	        if self.first and self.pre.val > root.val:
	            self.second = root
	        self.pre = root
	        self.inorder(root.right)
            	      
			
### 运行结果


	Runtime: 61 ms, faster than 85.84% of Python online submissions for Recover Binary Search Tree.
	Memory Usage: 13.9 MB, less than 52.31% of Python online submissions for Recover Binary Search Tree.

### 原题链接



https://leetcode.com/problems/recover-binary-search-tree/


您的支持是我最大的动力
