leetcode 1129. Shortest Path with Alternating Colors
 （python）




### 描述

You are given an integer n, the number of nodes in a directed graph where the nodes are labeled from 0 to n - 1. Each edge is red or blue in this graph, and there could be self-edges and parallel edges. You are given two arrays redEdges and blueEdges where:

* redEdges[i] = [ai, bi] indicates that there is a directed red edge from node ai to node bi in the graph, and
* blueEdges[j] = [uj, vj] indicates that there is a directed blue edge from node uj to node vj in the graph.

Return an array answer of length n, where each answer[x] is the length of the shortest path from node 0 to node x such that the edge colors alternate along the path, or -1 if such a path does not exist.





Example 1:

	Input: n = 3, redEdges = [[0,1],[1,2]], blueEdges = []
	Output: [0,1,-1]

	
Example 2:

	Input: n = 3, redEdges = [[0,1]], blueEdges = [[2,1]]
	Output: [0,1,-1]



Note:

* 	1 <= n <= 100
* 	0 <= redEdges.length, blueEdges.length <= 400
* 	redEdges[i].length == blueEdges[j].length == 2
* 	0 <= ai, bi, uj, vj < n

### 解析

根据题意，给定一个整数 n 是有向图中节点数，其中节点标记为从 0 到 n - 1 。在此图中，每条边都是红色或蓝色，并且可能存在自环和平行边。给定两个数组 redEdges 和 blueEdge，其中：

* redEdges[i] = [ai， bi] 表示图中存在从节点 ai 到节点 bi 的有向红边
* blueEdges[j] = [uj， vj] 表示图中存在从节点 uj 到节点 vj 的有向蓝色边

返回长度为 n 的数组 answer ，其中每个 answer[x] 是从节点 0 到节点 x 的红色边和蓝色边交替出现的最短路径的长度。如果此类路径不存在，则返回 -1 。

### 解答



### 运行结果



### 原题链接


https://leetcode.com/problems/shortest-path-with-alternating-colors/

您的支持是我最大的动力
