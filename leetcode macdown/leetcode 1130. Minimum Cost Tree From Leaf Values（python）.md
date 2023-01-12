leetcode  1130. Minimum Cost Tree From Leaf Values（python）

### 描述

Given an array arr of positive integers, consider all binary trees such that:

* Each node has either 0 or 2 children;
* The values of arr correspond to the values of each leaf in an in-order traversal of the tree.
* The value of each non-leaf node is equal to the product of the largest leaf value in its left and right subtree, respectively.

Among all possible binary trees considered, return the smallest possible sum of the values of each non-leaf node. It is guaranteed this sum fits into a 32-bit integer.

A node is a leaf if and only if it has zero children.



Example 1:

![](https://assets.leetcode.com/uploads/2021/08/10/tree1.jpg)

	Input: arr = [6,2,4]
	Output: 32
	Explanation: There are two possible trees shown.
	The first has a non-leaf node sum 36, and the second has non-leaf node sum 32.


	
Example 2:

![](https://assets.leetcode.com/uploads/2021/08/10/tree2.jpg)

	Input: arr = [4,11]
	Output: 44






Note:

	2 <= arr.length <= 40
	1 <= arr[i] <= 15
	It is guaranteed that the answer fits into a 32-bit signed integer (i.e., it is less than 2^31).


### 解析


根据题意，给定一个正整数数组 arr ，考虑所有二叉树，使得：

* 每个节点有 0 或 2 个孩子
* arr 的值对应于树的中序遍历中每个叶子的值
* 每个非叶节点的值分别等于其左右子树中最大叶值的乘积

可能会有多种二叉树，题目要求我们返回所有非叶节点的值的最小可能总和值。 题目会保证该总和在 32 位整数以内。一个节点是叶子当且仅当它有零个孩子。

首先我们通过简单的观察案例发现，要尽量把大的值放到深度较浅的叶子结点，把小的值放到深度较深的叶子结点，这样最后乘法的结果才会尽量小。一般有三种情况，我们举三种简单情况的例子：

* 例子一：[1,2,3] ，相乘的方法为 1 和 2 先乘得 2 ，然后 2 和 3 相乘，即从左往右操作
* 例子二：[3,2,1] ，相乘的方法为 1 和 2 先乘得 2 ，然后 2 和 3 相乘，即从右往左操作
* 例子三：[3,1,2] ，取 3 和 2 的较小值为 2 ，先对 1 和 2 相乘得 2 ，再 2 和 3 相乘

因为题目中对值的限制在 1 到 15 之间，在例子一前面加一个超大值 100 ，就和例子三一样，所以一共有两种情况，我们维护一个栈 stack 在最前面加一个超大值 100 ，先按照例子二的逻辑计算乘积和，再按照例子二的逻辑计算乘积和，更详细的解释可以看大神[视频详解](https://www.bilibili.com/video/BV1At411c7Dc)


### 解答
				
	
	class Solution(object):
	    def mctFromLeafValues(self, arr):
	        """
	        :type arr: List[int]
	        :rtype: int
	        """
	        stack = [100]
	        result = 0
	        for current in arr:
	            while stack[-1]<=current:
	                drop = stack.pop()
	                result += drop*(min(stack[-1], current))
	            stack.append(current)
	        while len(stack)>2:
	            result += stack.pop() * stack[-1]
	        return result
	            
	       	      
			
### 运行结果

	Runtime: 28 ms, faster than 46.53% of Python online submissions for Minimum Cost Tree From Leaf Values.
	Memory Usage: 13.4 MB, less than 74.26% of Python online submissions for Minimum Cost Tree From Leaf Values.


原题链接：https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/



您的支持是我最大的动力
