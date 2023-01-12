leetcode  1631. Path With Minimum Effort（python）




### 描述


You are a hiker preparing for an upcoming hike. You are given heights, a 2D array of size rows x columns, where heights[row][col] represents the height of cell (row, col). You are situated in the top-left cell, (0, 0), and you hope to travel to the bottom-right cell, (rows-1, columns-1) (i.e., 0-indexed). You can move up, down, left, or right, and you wish to find a route that requires the minimum effort.

A route's effort is the maximum absolute difference in heights between two consecutive cells of the route.

Return the minimum effort required to travel from the top-left cell to the bottom-right cell.


Example 1:

![](https://assets.leetcode.com/uploads/2020/10/04/ex1.png)


	Input: heights = [[1,2,2],[3,8,2],[5,3,5]]
	Output: 2
	Explanation: The route of [1,3,5,3,5] has a maximum absolute difference of 2 in consecutive cells.
	This is better than the route of [1,2,2,2,5], where the maximum absolute difference is 3.

	
Example 2:


![](https://assets.leetcode.com/uploads/2020/10/04/ex2.png)

	Input: heights = [[1,2,3],[3,8,4],[5,3,5]]
	Output: 1
	Explanation: The route of [1,2,3,4,5] has a maximum absolute difference of 1 in consecutive cells, which is better than route [1,3,5,3,5].

Example 3:


![](https://assets.leetcode.com/uploads/2020/10/04/ex3.png)

	Input: heights = [[1,2,1,1,1],[1,2,1,2,1],[1,2,1,2,1],[1,2,1,2,1],[1,1,1,2,1]]
	Output: 0
	Explanation: This route does not require any effort.
	



Note:

	rows == heights.length
	columns == heights[i].length
	1 <= rows, columns <= 100
	1 <= heights[i][j] <= 10^6


### 解析


根据题意， 给定大小为 rows x columns 的二维数组 heights，其中 heights[row][col] 表示山的高度 (row, col)。 初始位于左上角的单元格 (0, 0)，然后前往右下角的单元格 (rows-1, columns-1) 。 允许向上、向下、向左或向右移动，最终找到一条需要最少努力的爬山路线。路线的努力程度是整个路线中某两个连续单元格之间的最大的绝对高度差。返回从左上角移动到右下角所需的最小努力值。

这两天的每日一题，考察的都是并查集的相关知识，代码背过了。这道题虽然是一个二维数组，但是可以拉成一维数组，这样就可以比较好的写代码了，如果不懂，可以先做一下昨天的每日一题 [1202. Smallest String With Swaps](https://leetcode.com/problems/smallest-string-with-swaps/) 。

这道题的关键点就是将题意转化为图，每个格子就是一个节点，而边的权重就是相邻格子的高度差绝对值，然后我们找从左上角到右下角的最少努力程度的路线。那么解题的关键就是成了如何找出最少努力的路线，那肯定就是并查集了，我们将所有的权重进行升序排序，，然后按照权重大小逐个将连通性补齐，当我们添加某个权重的边之后，左上角和右下角的节点完全连通，说明这个权重大小就是我们要找的答案，当 heights 只有一个格子的时候，这是一种边界情况直接返回 0 即可。

时间复杂度为 O(NlogN)  这里的 N 是 M\*N 因为我们将二维数组拉成了一维数组，空间复杂度为 O(M\*N)。

### 解答
				
	class Solution(object):
	    def minimumEffortPath(self, heights):
	        """
	        :type heights: List[List[int]]
	        :rtype: int
	        """
	        M = len(heights)
	        N = len(heights[0])
	
	        father = [0] * (M*N)
	        for i in range(M):
	            for j in range(N):
	                father[i*N + j] = i*N + j
	
	        def findFater(x):
	            if father[x]!=x:
	                father[x] = findFater(father[x])
	            return father[x]
	
	        def union(x,y):
	            x = findFater(x)
	            y = findFater(y)
	            if y > x:
	                father[y] = x
	            else:
	                father[x] = y
	        edges = []
	        for i in range(M):
	            for j in range(N):
	                idx = i * N + j
	                if i < M - 1:
	                    edges.append([abs(heights[i+1][j]-heights[i][j]), idx, idx + N])
	                if j < N - 1:
	                    edges.append([abs(heights[i][j+1]-heights[i][j]), idx, idx + 1])
	        edges.sort()
	        for w,i,j in edges:
	            if findFater(i) != findFater(j):
	                union(i, j)
	            if findFater(0) == findFater(M*N-1):
	                return w
	        return 0

            	      
			
### 运行结果


	Runtime: 840 ms, faster than 82.52% of Python online submissions for Path With Minimum Effort.
	Memory Usage: 17.3 MB, less than 20.39% of Python online submissions for Path With Minimum Effort.

### 原题链接


https://leetcode.com/problems/path-with-minimum-effort/


您的支持是我最大的动力
