leetcode  2421. Number of Good Paths（python）




### 描述

There is a tree (i.e. a connected, undirected graph with no cycles) consisting of n nodes numbered from 0 to n - 1 and exactly n - 1 edges. You are given a 0-indexed integer array vals of length n where vals[i] denotes the value of the i<sup>th</sup> node. You are also given a 2D integer array edges where edges[i] = [a<sub>i</sub>, b<sub>i</sub>] denotes that there exists an undirected edge connecting nodes a<sub>i</sub> and b<sub>i</sub>.

A good path is a simple path that satisfies the following conditions:

* The starting node and the ending node have the same value.
* All nodes between the starting node and the ending node have values less than or equal to the starting node (i.e. the starting node's value should be the maximum value along the path).

Return the number of distinct good paths.

Note that a path and its reverse are counted as the same path. For example, 0 -> 1 is considered to be the same as 1 -> 0. A single node is also considered as a valid path.



Example 1:

![](https://assets.leetcode.com/uploads/2022/08/04/f9caaac15b383af9115c5586779dec5.png)

	Input: vals = [1,3,2,1,3], edges = [[0,1],[0,2],[2,3],[2,4]]
	Output: 6
	Explanation: There are 5 good paths consisting of a single node.
	There is 1 additional good path: 1 -> 0 -> 2 -> 4.
	(The reverse path 4 -> 2 -> 0 -> 1 is treated as the same as 1 -> 0 -> 2 -> 4.)
	Note that 0 -> 2 -> 3 is not a good path because vals[2] > vals[0].

	
Example 2:


![](https://assets.leetcode.com/uploads/2022/08/04/149d3065ec165a71a1b9aec890776ff.png)

	Input: vals = [1,1,2,2,3], edges = [[0,1],[1,2],[2,3],[2,4]]
	Output: 7
	Explanation: There are 5 good paths consisting of a single node.
	There are 2 additional good paths: 0 -> 1 and 2 -> 3.

Example 3:

![](https://assets.leetcode.com/uploads/2022/08/04/31705e22af3d9c0a557459bc7d1b62d.png)

	Input: vals = [1], edges = []
	Output: 1
	Explanation: The tree consists of only one node, so there is one good path.

Note:

* 	n == vals.length
* 	1 <= n <= 3 * 10^4
* 	0 <= vals[i] <= 10^5
* 	edges.length == n - 1
* 	edges[i].length == 2
* 	0 <= a<sub>i</sub>, b<sub>i</sub> < n
* 	a<sub>i</sub> != b<sub>i</sub>
* 	edges represents a valid tree.


### 解析

根据题意，有一棵树由 n 个节点组成，编号从 0 到 n - 1 ，正好有 n - 1 条边。给定一个长度为 n 的 0 索引的整数数组 vals  ，其中 vals[i] 表示第 i 个节点的值。另外给出一个二维整数数组 edges ，其中 edges[i] = [a<sub>i</sub>, b<sub>i</sub>] 表示存在连接节点 a<sub>i</sub> 和 b<sub>i</sub> 的无向边。

一条好的路径得满足以下条件：

* 起始节点和结束节点具有相同的值。
* 起始节点和结束节点之间的所有节点的值都小于或等于起始节点，即起始节点的值应为路径上的最大值。

返回不同良好路径的数量。请注意，路径与其反向路径被认为是同一路径。例如，0 -> 1 被视为与 1 -> 0 相同。单个节点也被视为有效路径。

### 解答



### 运行结果



### 原题链接

https://leetcode.com/problems/number-of-good-paths/description/


您的支持是我最大的动力
