leetcode  1104. Path In Zigzag Labelled Binary Tree（python）

### 描述


In an infinite binary tree where every node has two children, the nodes are labelled in row order.

In the odd numbered rows (ie., the first, third, fifth,...), the labelling is left to right, while in the even numbered rows (second, fourth, sixth,...), the labelling is right to left.

![](https://assets.leetcode.com/uploads/2019/06/24/tree.png)

Given the label of a node in this tree, return the labels in the path from the root of the tree to the node with that label.

Example 1:


	Input: label = 14
	Output: [1,3,4,14]
	
Example 2:


	Input: label = 26
	Output: [1,2,6,10,26]




Note:


	1 <= label <= 10^6

### 解析


根据题意，给出了一个无限满二叉树，所有的节点都被从 1 开始的整数打了标签，而且在树的奇数行都是正常的从左到右排列数字，但在树的偶数行都是从右到左的顺序排列数字，现在题目给出了一个目标数字 label ，让我们在这棵树中找出从根节点 root 到 label 的节点列表。

其实这道题就是一个找规律的题，只是套在了树结构上。如果题中的树是一棵节点数字正常的从左到右的树，那么可以看出来知道目标节点 label 之后，倒推其父节点都是 label// 2 ，一直到根节点 1 为止，假如目标 label 为 7 ，则结果为 [1,3,7] , 树结构如下：

			1
		2		3
	  4   5   6	  7
	  
 但是现在的树在偶数行上是反转排列的，树结构如下所示：
 
 			1
		3		2
	  4   5   6	  7
 
 
 所以我们只要知道从 label 到 root 经过的节点上，原来的位置上的数字是什么就可以解决此题，通过找规律我们发现目标不管在偶数行还是奇数行，某个标签 label 的原位置上的数字为 
  
 	2 ** (level) - 1 + 2 ** (level-1) - label 

level 为当前节点的深度，label 为当前节点的值，算出来即为当前位置的原值，对其除 2 取整数即可知道其父节点值为多少。不断重复这个找当前位置的原值、以及除 2 取整数得到父节点的操作，直到层数到达第一层为之，然后将找到的所有的节点升序排列放入列表 result 中返回即可。

### 解答
				
	class Solution(object):
	    def pathInZigZagTree(self, label):
	        """
	        :type label: int
	        :rtype: List[int]
	        """
	        result = [label]
	        level = int(math.log(label, 2)) + 1
	        while level>1:
	            origin = 2 ** (level) - 1 + 2 ** (level-1) - label
	            label = origin // 2
	            result.append(label)
	            level -= 1
	        result = result[::-1]
	        return result

            	      
			
### 运行结果

	Runtime: 12 ms, faster than 92.11% of Python online submissions for Path In Zigzag Labelled Binary Tree.
	Memory Usage: 13.4 MB, less than 63.16% of Python online submissions for Path In Zigzag Labelled Binary Tree.


### 解析

这种方法是在论坛里面看的一种方法，我们的结果其实就是可以看作从 label 到根节点 root 的过程，如题中的目标 label 为 14 可以看成二进制 1110 ，其在树上的父节点为 4（100） ，4 的父节点为 3（11），3 的父节点为 1（1）可以发现规律就是父节点的值的二进制和当前节点的二进制存在关系：

	parent = 1 + inverted(label[1:-1])

inverted 为函数，将 1 变为 0 ，将 0 变为 1 。一直重复这个计算过程，根据这个规律便可以解题。
### 解答

	class Solution(object):
	    def pathInZigZagTree(self, label):
	        """
	        :type label: int
	        :rtype: List[int]
	        """
	        res = []
	        while label != 1:
	            res.append(label)
	            label = int('1' + "".join(map(lambda x: '1' if x == '0' else '0', format(label, 'b')[1:-1])), 2)
	        res.append(1)
	        return reversed(res)

### 运行结果

	Runtime: 37 ms, faster than 21.05% of Python online submissions for Path In Zigzag Labelled Binary Tree.
	Memory Usage: 13.3 MB, less than 63.16% of Python online submissions for Path In Zigzag Labelled Binary Tree.

原题链接：https://leetcode.com/problems/path-in-zigzag-labelled-binary-tree/



您的支持是我最大的动力
