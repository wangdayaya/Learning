leetcode  606. Construct String from Binary Tree（python）




### 描述

Given the root of a binary tree, construct a string consisting of parenthesis and integers from a binary tree with the preorder traversal way, and return it.

Omit all the empty parenthesis pairs that do not affect the one-to-one mapping relationship between the string and the original binary tree.



Example 1:

![](https://assets.leetcode.com/uploads/2021/05/03/cons1-tree.jpg)

	Input: root = [1,2,3,4]
	Output: "1(2(4))(3)"
	Explanation: Originally, it needs to be "1(2(4)())(3()())", but you need to omit all the unnecessary empty parenthesis pairs. And it will be "1(2(4))(3)"

	
Example 2:

![](https://assets.leetcode.com/uploads/2021/05/03/cons2-tree.jpg)

	Input: root = [1,2,3,null,4]
	Output: "1(2()(4))(3)"
	Explanation: Almost the same as the first example, except we cannot omit the first parenthesis pair to break the one-to-one mapping relationship between the input and the output.




Note:

	The number of nodes in the tree is in the range [1, 10^4].
	-1000 <= Node.val <= 1000


### 解析

根据题意，给定二叉树的根 root ，用前序遍历的方式从二叉树构造一个由括号和整数组成的字符串并返回。这里有一个需要注意的点，题目要求我们省略所有不影响字符串与原二叉树一一映射关系的空括号对。像例子一中描述的那样，我们最开始得到的结果应该是 1(2(4)())(3()()) ，但是因为里面的空括号对移除之后也不会影响字符串与二叉树的对应关系，所以最后就需要去掉 1(2(4)())(3()()) 中的空括号对得到最后的  1(2(4))(3) 。但是像例子二就没有办法移除，因为移除之后会破坏一一对应关系。

通过观察其实我们可以发现省略括号的情况有两种：

* 当左子树和右子树都为空的时候，我们可以将它们代表的两个空括号对都进行省略，不会影响题目一一对应的要求
* 当左子树不为空，但是右子树为空的时候，我们可以将右子树的空括号对进行省略，不会影响题目一一对应的要求

有了上面的结论，我们针对树类型的题目，直接使用 DFS 最方便，我们定义函数 tree2str 为当前以 root 为根的子树所能形成的字符串，递归逻辑如下：

* 当 root 为空的时候表示递归到了空叶子结点，直接返回空字符串即可
* 然后递归生成左子树的字符串 l ，和递归生成右子树的字符串 r
* 然后根据上面的两个结论，将 l 或者 r 中可以省略的部分换成空字符串
* 最后将根节点字符和 l 、r 进行拼接，然后返回即可

时间复杂度为 O(N) ，N 就是二叉树的节点个数，其实就是遍历所有节点的时间，空间复杂度为 O(N) ，因为最坏的情况下要递归 N 层。

### 解答

	class Solution:
	    def tree2str(self, root: Optional[TreeNode]) -> str:
	        if not root:
	            return ''
	        result = str(root.val)
	        l = '(' + self.tree2str(root.left) + ')'
	        r =  '(' + self.tree2str(root.right) + ')'
	        if l!= '()' and r == '()':
	            r = ''
	        if l == '()' and r =='()':
	            l = r =''
	        result = result + l + r
	        return result

### 运行结果

	Runtime: 48 ms, faster than 85.96% of Python online submissions for Construct String from Binary Tree.
	Memory Usage: 16.2 MB, less than 49.25% of Python online submissions for Construct String from Binary Tree.

### 原题链接


https://leetcode.com/problems/construct-string-from-binary-tree/

您的支持是我最大的动力
