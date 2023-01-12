leetcode  116. Populating Next Right Pointers in Each Node（python）

### 每日经典

《南园十三首·其五》 ——李贺（唐）

男儿何不带吴钩，收取关山五十州。

请君暂上凌烟阁，若个书生万户侯？


### 描述


You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:

	struct Node {
	  int val;
	  Node *left;
	  Node *right;
	  Node *next;
	}

Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

Follow-up:

* You may only use constant extra space.
* The recursive approach is fine. You may assume implicit stack space does not count as extra space for this problem.


Example 1:


![](https://assets.leetcode.com/uploads/2019/02/14/116_sample.png)

	Input: root = [1,2,3,4,5,6,7]
	Output: [1,#,2,3,#,4,5,6,7,#]
	Explanation: Given the above perfect binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.
	
Example 2:

	Input: root = []
	Output: []



Note:

	The number of nodes in the tree is in the range [0, 2^12 - 1].
	-1000 <= Node.val <= 1000



### 解析


根据题意，给定一个完美的二叉树，其中所有叶子都在同一层上，每个父节点都有两个子节点，同时给出了每个节点的结构定义，每个节点的 next 指针指向其同层的右节点。如果没有下一个右节点，则 next 指针设置为 NULL 。最初，所有 next 指针都设置为 NULL。题目还为高水平选手提出了更高的要求：

* 只能使用常数级别的额外空间
* 使用递归方法

这道题乍一看不知道怎么入手，但是摸清楚规律也是很好解决的，首先我觉得只要是树的题目，基本上 99% 就能用 DFS 解题，这道题也不例外。这道题有三个关键点：

* 第一个是如何判断最右边的节点 next 为 None ，其实通过例子我们可以看出来只要其父节点的 next 为 None ，它的 next 就为 None 。
* 第二个关键点就是当右孩子右边还有节点，其 next 指向哪里，从例子中我们同样可以找到，就是其父节点的 next 指向节点的 left 节点。
* 第三个问题是递归什么时候停止，那就是当节点为空或者没有下一层节点的时候，没有下一层节点很好判断，直接判断某个节点的 left 为空即可。

解决了这三个关键的问题，定义 DFS 函数，表示以当前节点为根节点的时候，为其左孩子和右孩子的 next 赋值。

### 解答
				

	class Node(object):
	    def __init__(self, val=0, left=None, right=None, next=None):
	        self.val = val
	        self.left = left
	        self.right = right
	        self.next = next
	
	class Solution(object):
	    def connect(self, root):
	        """
	        :type root: Node
	        :rtype: Node
	        """
	        self.dfs(root)
	        return root
	        
	    def dfs(self, root):
	        if not root or not root.left: return 
	        root.left.next = root.right
	        if root.next:
	            root.right.next = root.next.left
	        else:
	            root.right.next = None
	        self.dfs(root.left)
	        self.dfs(root.right)
	            
	        
	        
            	      
			
### 运行结果


	Runtime: 97 ms, faster than 6.41% of Python online submissions for Populating Next Right Pointers in Each Node.
	Memory Usage: 16.2 MB, less than 92.50% of Python online submissions for Populating Next Right Pointers in Each Node.


### 解析

从题目中节点的 next 的指向顺序是从左到右这一特点中我们可以联想到用 BFS 的思想解题。整体思路和上面一样，从根节点开始，将每个遍历到的节点都放入 stack 中，然后对其左节点和右节点的 next 进行处理，这里使用 stack.pop() 和  stack.pop(0)  都可以，只不过是遍历节点的顺序不同，不影响对每个节点的 next 的赋值。

### 解答

	class Node(object):
	    def __init__(self, val=0, left=None, right=None, next=None):
	        self.val = val
	        self.left = left
	        self.right = right
	        self.next = next
	
	class Solution(object):
	    def connect(self, root):
	        """
	        :type root: Node
	        :rtype: Node
	        """
	        stack = [root]
	        while stack:
	            node = stack.pop()
	            if node and node.left:
	                node.left.next = node.right
	                if node.next:
	                    node.right.next = node.next.left
	                else:
	                    node.right.next = None
	                stack.append(node.left)
	                stack.append(node.right)
	        return root
	                
	        
	        
	                
	        
	        

### 运行结果

	Runtime: 52 ms, faster than 77.19% of Python online submissions for Populating Next Right Pointers in Each Node.
	Memory Usage: 16.5 MB, less than 47.85% of Python online submissions for Populating Next Right Pointers in Each Node.


原题链接：https://leetcode.com/problems/populating-next-right-pointers-in-each-node/



您的支持是我最大的动力
