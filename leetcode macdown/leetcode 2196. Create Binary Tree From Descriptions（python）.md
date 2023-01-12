leetcode  2196. Create Binary Tree From Descriptions（python）


### 前言

这是 Weekly Contest 283 的第三道题，考察的是二叉树的构建，同时巧用字典数据结构来解题，难度为 Medium 。

### 描述


You are given a 2D integer array descriptions where descriptions[i] = [parent<sub>i</sub>, child<sub>i</sub>, isLeft<sub>i</sub>] indicates that parent<sub>i</sub> is the parent of child<sub>i</sub> in a binary tree of unique values. Furthermore,

* If isLeft<sub>i</sub> == 1, then child<sub>i</sub> is the left child of parent<sub>i</sub>.
* If isLeft<sub>i</sub> == 0, then child<sub>i</sub> is the right child of parent<sub>i</sub>.

Construct the binary tree described by descriptions and return its root.

The test cases will be generated such that the binary tree is valid.


Example 1:

![](https://assets.leetcode.com/uploads/2022/02/09/example1drawio.png)

	Input: descriptions = [[20,15,1],[20,17,0],[50,20,1],[50,80,0],[80,19,1]]
	Output: [50,20,80,15,17,19]
	Explanation: The root node is the node with value 50 since it has no parent.
	The resulting binary tree is shown in the diagram.

	





Note:

* 	1 <= descriptions.length <= 10^4
* 	descriptions[i].length == 3
* 	1 <= parent<sub>i</sub>, child<sub>i</sub> <= 10^5
* 	0 <= isLeft<sub>i</sub> <= 1
* 	The binary tree described by descriptions is valid.


### 解析


根据题意，给出了一个二维数组 descriptions ，其中 descriptions[i] = [parent<sub>i</sub>, child<sub>i</sub>, isLeft<sub>i</sub>] ，表示的 parent<sub>i</sub> 是 child<sub>i</sub> 父节点，并且当 isLeft<sub>i</sub> 为 1 的时候 child<sub>i</sub> 是 parent<sub>i</sub> 的左节点，当  isLeft<sub>i</sub> 为 0 的时候  child<sub>i</sub> 是 parent<sub>i</sub> 的右节点。

我们要做的就是根据这个 descriptions 描述，将这个二叉树重新构造出来，并且返回根节点。另外题目保证了给出的测试用例都是合法的。

因为 descriptions 的长度最大为 10^4 ，而且树节点的个数最大为  10^5 ，所以只能用 O(N) 的解法来解题，否则会超时，我们这里用的是用空间换时间，将每个节点都保存在一个字典 d 中，每个节点的值对应起节点对象。然后我们从左到右遍历 descriptions 中的子数组 [p,c,flag]：

* 如果当前 p 没在字典 d 中，我们就新建一个节点对象 TreeNode(p) ，并将其赋与 d[p] ；
* 如果 c 没在字典中，同样新建一个节点对象 TreeNode(c) ，并将其赋与 d[c]； 
* 如果 flag 为 1 ，则将 d[c] 赋与 d[p].left 指针，否则将  d[c] 赋与 d[p].right 指针 

遍历的同时我们将 c 放入一个集合 s 之中，s 中的元素都是当过子节点的元素，等遍历结束，d 字典中的所有 key 组成的集合都是当过父节点的元素，根节点肯定是只做父节点不做子节点的元素，我们用字典 d 的所有 key 的集合与 s 作差集就能得到根节点的值 result ，最后返回 d[result] 即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答
				

	class Solution(object):
	    def createBinaryTree(self, descriptions):
	        d = {}
	        s = set()
	        for p,c,flag in descriptions:
	            s.add(c)
	            if p not in d:
	                d[p] = TreeNode(p)
	            if c not in d:
	                d[c] = TreeNode(c)
	            if flag == 1:
	                d[p].left = d[c]
	            else:
	                d[p].right = d[c]
	        result = list(set(d.keys())-s)
	        print(result)
	        return d[result[0]]
			
### 运行结果

	85 / 85 test cases passed.
	Status: Accepted
	Runtime: 2445 ms
	Memory Usage: 29.7 MB


### 原题链接


https://leetcode.com/contest/weekly-contest-283/problems/create-binary-tree-from-descriptions/


您的支持是我最大的动力
