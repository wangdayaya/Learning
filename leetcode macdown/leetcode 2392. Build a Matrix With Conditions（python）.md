leetcode  2392. Build a Matrix With Conditions（python）




### 描述



You are given a positive integer k. You are also given:

* a 2D integer array rowConditions of size n where rowConditions[i] = [above<sub>i</sub>, below<sub>i</sub>], and
* a 2D integer array colConditions of size m where colConditions[i] = [left<sub>i</sub>, right<sub>i</sub>].

The two arrays contain integers from 1 to k. You have to build a k x k matrix that contains each of the numbers from 1 to k exactly once. The remaining cells should have the value 0. The matrix should also satisfy the following conditions:

* The number above<sub>i</sub> should appear in a row that is strictly above the row at which the number below<sub>i</sub> appears for all i from 0 to n - 1.
* The number left<sub>i</sub> should appear in a column that is strictly left of the column at which the number right<sub>i</sub> appears for all i from 0 to m - 1.

Return any matrix that satisfies the conditions. If no answer exists, return an empty matrix.

Example 1:

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f0901e9a98d74255ad86a67902c9bc96~tplv-k3u1fbpfcp-zoom-1.image)

	Input: k = 3, rowConditions = [[1,2],[3,2]], colConditions = [[2,1],[3,2]]
	Output: [[3,0,0],[0,0,1],[0,2,0]]
	Explanation: The diagram above shows a valid example of a matrix that satisfies all the conditions.
	The row conditions are the following:
	- Number 1 is in row 1, and number 2 is in row 2, so 1 is above 2 in the matrix.
	- Number 3 is in row 0, and number 2 is in row 2, so 3 is above 2 in the matrix.
	The column conditions are the following:
	- Number 2 is in column 1, and number 1 is in column 2, so 2 is left of 1 in the matrix.
	- Number 3 is in column 0, and number 2 is in column 1, so 3 is left of 2 in the matrix.
	Note that there may be multiple correct answers.

	
Example 2:

	Input: k = 3, rowConditions = [[1,2],[2,3],[3,1],[2,3]], colConditions = [[2,1]]
	Output: []
	Explanation: From the first two conditions, 3 has to be below 1 but the third conditions needs 3 to be above 1 to be satisfied.
	No matrix can satisfy all the conditions, so we return the empty matrix.




Note:

* 	2 <= k <= 400
* 	1 <= rowConditions.length, colConditions.length <= 10^4
* 	rowConditions[i].length == colConditions[i].length == 2
* 	1 <= above<sub>i</sub>, below<sub>i</sub>, left<sub>i</sub>, right<sub>i</sub> <= k
* 	above<sub>i</sub> != below<sub>i</sub>
* 	left<sub>i</sub> != right<sub>i</sub>


### 解析

给定一个正整数 k ，还给出：

* 一个大小为 n 的 2D 整数数组 rowConditions，其中 rowConditions[i] = [above<sub>i</sub>, below<sub>i</sub>] 和
* 一个大小为 m 的二维整数数组 colConditions，其中 colConditions[i] = [left<sub>i</sub>, right<sub>i</sub>]。

这两个数组包含从 1 到 k 的整数。 我们必须构建一个 k x k 矩阵，其中包含从 1 到 k 的每个数字恰好一次。 其余单元格的值应为 0。矩阵还应满足以下条件：

* 在矩阵的所有行中，数字 above<sub>i</sub> 应必须出现在数字 below<sub>i</sub> 的上面。
* 在矩阵的所有列中，数字 left<sub>i</sub> 应必须出现在数字 right<sub>i</sub> 的左面。

返回任何满足条件的矩阵。 如果不存在答案，则返回一个空矩阵。

其实这道题中没有要求行与列之间的关系，所以行和列可以单独分别去进行拓扑排序，所以其实最关键的地方就是我们定义的拓扑函数，逻辑如下：

- 首先要定义一个图 graph ，并且找出每个节点的入度 indegree ，我们遍历 rowConditions ／ colConditions 要将其转换为 graph 的形式，并且每个节点的入度都记录在 indegree ，然后将此时入度为 0 的节点都放入队列 deque ，没有入度意味着没有前驱，要最先开始进行执行。

- 循环弹出 deque 的最左的元素，将其记录道结果 result 中，并且遍历 x 的后续节点，因为 x 已经遍历过，所以 x 的后续节点的 indegree 都要减一，如果此时某个后续节点的入度减为 0 则将其加入 deque 最后，等待弹出

- 等到循环结束，如果 result 的个数和 k 一样说明没有环直接返回 result 即可，否则说明有环返回空列表

我们对行与列分别进行拓扑排序之后，然后将 1-k 之间的每个元素 n 在行和列各自的拓扑顺序中找出对应的索引的位置组成 [i,j] ，那么就将 n 填充在 result[i][j] 的位置上即可，其他位置填充 0 ，最后将 result 返回即可。

时间复杂度为 O(k^2) ，空间复杂度为 O(k^2) 。

### 解答

	class Solution(object):
	    def buildMatrix(self, k, rowConditions, colConditions):
	        def topo(A):
	            graph = [[] for _ in range(k)]
	            indegree = [0] * k
	            result = []
	            for x,y in A:
	                graph[x-1].append(y-1)
	                indegree[y-1] += 1
	            deque = collections.deque(i for i,v in enumerate(indegree) if v==0)
	            while deque:
	                x = deque.popleft()
	                result.append(x)
	                for nxt in graph[x]:
	                    indegree[nxt] -= 1
	                    if indegree[nxt] == 0:
	                        deque.append(nxt)
	            return result if len(result)==k else []
	
	        row, col = topo(rowConditions), topo(colConditions)
	        if not row or not col:
	            return []
	        result = [[0]*k for _ in range(k)]
	        for i in range(k):
	            result[row.index(i)][col.index(i)] = i+1
	        return result

### 运行结果

	51 / 51 test cases passed.
	Status: Accepted
	Runtime: 977 ms
	Memory Usage: 21.9 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-308/problems/build-a-matrix-with-conditions/


您的支持是我最大的动力