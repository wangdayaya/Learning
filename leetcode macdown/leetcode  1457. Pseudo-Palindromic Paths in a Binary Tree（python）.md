leetcode  1457. Pseudo-Palindromic Paths in a Binary Tree（python）




### 描述

Given a binary tree where node values are digits from 1 to 9. A path in the binary tree is said to be pseudo-palindromic if at least one permutation of the node values in the path is a palindrome. Return the number of pseudo-palindromic paths going from the root node to leaf nodes.





Example 1:

![](https://assets.leetcode.com/uploads/2020/05/06/palindromic_paths_1.png)

	Input: root = [2,3,1,3,1,null,1]
	Output: 2 
	Explanation: The figure above represents the given binary tree. There are three paths going from the root node to leaf nodes: the red path [2,3,3], the green path [2,1,1], and the path [2,3,1]. Among these paths only red path and green path are pseudo-palindromic paths since the red path [2,3,3] can be rearranged in [3,2,3] (palindrome) and the green path [2,1,1] can be rearranged in [1,2,1] (palindrome).

	
Example 2:

![](https://assets.leetcode.com/uploads/2020/05/07/palindromic_paths_2.png)

	Input: root = [2,1,1,1,3,null,null,null,null,null,1]
	Output: 1 
	Explanation: The figure above represents the given binary tree. There are three paths going from the root node to leaf nodes: the green path [2,1,1], the path [2,1,3,1], and the path [2,1]. Among these paths only the green path is pseudo-palindromic since [2,1,1] can be rearranged in [1,2,1] (palindrome).


Example 3:


	Input: root = [9]
	Output: 1


Note:

	The number of nodes in the tree is in the range [1, 10^5].
	1 <= Node.val <= 9


### 解析

根据题意，给定一棵二叉树，其中节点值是从 1 到 9 的数字。如果路径中的节点值能变成一个回文序列，则称这条路径为二叉树中的伪回文路径。 返回从根节点到叶节点的伪回文路径数。

解决这道题的核心是要理解回文序列的定义，其实回文序列有两种形式，如下所示：

* 以中间的单个数字成镜像分布，如“12321”。
* 以中间的两个相同的数字成镜像分布，如“123321”。

所以本质上我们只要用一种算法知道整个序列是以中间一个或者两个相同的字母形成镜像对称即可，我们在这里要使用一种巧妙的位异或运算来解决这个问题，首先我们要知道这里总共只有 1-9 的这几种数字，我们要使用位运算之前先把每个数字转化成对应的二进制表示如 1 可以转化为 1<<1 （也就是二进制 10） ，2 可以转化为 1<<2（也就是二进制 100 ） ，以此类推，这样最后能把这些数字都转化为二进制数字，然后使用这些转化后的数字进行异或运算。

因为异或运算是“二进制位上相同为 0 ，不同为 1 ”的运算，所以回文序列中的所有数字经过异或运算之后，会有两种结果：

* 针对上面第一种情况，因为把成对的数字都经过计算抵消为 0 了，结果只会剩下一个序列中的奇数个数的那个数字，这里有一个位运算技巧，总所周知 n & (n - 1) 可以用来消除最后一个 1 ，所以最后如果判断这个 n & (n - 1) 为 0 即可 。
* 针对上面第二种情况，结果为 0 ，因为数字都两两相同 。


思路清晰之后，我们定义递归函数 dfs ，表示在以 root 为开始并且以叶子结点为结束的值的某条序列当中，如果最后这个序列中的所有值经过异或运算的结果 temp 变为 0 或者变为 temp & (temp-1) == 0 ，那么说明这条序列是可以形成伪回文序列的，只需要将结果 result 加一即可，递归结束只需要返回 result 即可。

N 为节点数量，时间复杂度为 O(N) ，空间复杂度为 O(N) ，可能递归 N 次。 





### 解答

	class Solution(object):
	    def __init__(self):
	        self.result = 0
	    def pseudoPalindromicPaths (self, root):
	        """
	        :type root: TreeNode
	        :rtype: int
	        """
	        def dfs(root,temp):
	            if root is None:
	                return None
	            temp ^= 1 << root.val
	            if root.left is None and root.right is None:
	                if temp == 0 or temp & (temp -1) == 0:
	                    self.result += 1
	            dfs(root.left, temp)
	            dfs(root.right, temp)
	        dfs(root, 0)
	        return self.result


### 运行结果

	Runtime: 1189 ms, faster than 90.32% of Python online submissions for Pseudo-Palindromic Paths in a Binary Tree.
	Memory Usage: 137.3 MB, less than 35.48% of Python online submissions for Pseudo-Palindromic Paths in a Binary Tree.

### 原题链接

https://leetcode.com/problems/pseudo-palindromic-paths-in-a-binary-tree/


您的支持是我最大的动力
