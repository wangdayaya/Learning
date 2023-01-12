leetcode  378. Kth Smallest Element in a Sorted Matrix（python）




### 描述

Given an n x n matrix where each of the rows and columns is sorted in ascending order, return the kth smallest element in the matrix. Note that it is the kth smallest element in the sorted order, not the kth distinct element. You must find a solution with a memory complexity better than O(n^2).



Example 1:


	Input: matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
	Output: 13
	Explanation: The elements in the matrix are [1,5,9,10,11,12,13,13,15], and the 8th smallest number is 13
	
Example 2:


	Input: matrix = [[-5]], k = 1
	Output: -5



Note:

	n == matrix.length == matrix[i].length
	1 <= n <= 300
	-10^9 <= matrix[i][j] <= 10^9
	All the rows and columns of matrix are guaranteed to be sorted in non-decreasing order.
	1 <= k <= n^2


### 解析

根据题意，给定一个 n x n 矩阵 matrix ，其中每一行和每一列都按升序排序，返回矩阵中第 k 个最小的元素。 请注意，它是排序顺序中的第 k 个最小元素，而不是第 k 个不同的元素。 而且题目要求找到一个空间复杂度优于 O(n^2) 的解决方案。

最笨的方法肯定是使用空间复杂度 O(n^2) 的方法，先都进行排序都一个有序列表，然后取找第 k 个元素。我们使用另一种方法，因为每一行是有序的，我们可以考虑每次从矩阵中弹出一个最小值，那么弹出的第 k 个元素一定是我们想要的结果，这里就要用到了小根堆数据结构。

时间复杂度位 O(klogN) ，如果 k 为 N^2 那么时间复杂度为 O(N^2logN)，空间复杂度为 O(N) ，
### 解答

	class Solution(object):
	    def kthSmallest(self, matrix, k):
	        """
	        :type matrix: List[List[int]]
	        :type k: int
	        :rtype: int
	        """
	        M = len(matrix)
	        N = len(matrix[0])
	        L = [(matrix[i][0],i,0) for i in range(M)]
	        heapq.heapify(L)
	        for _ in range(k-1):
	            _, x,y = heapq.heappop(L)
	            if y!=N-1:
	                heapq.heappush(L, (matrix[x][y+1], x, y+1))
	        return heapq.heappop(L)[0]

### 运行结果

	Runtime: 293 ms, faster than 51.00% of Python online submissions for Kth Smallest Element in a Sorted Matrix.
	Memory Usage: 17.4 MB, less than 85.38% of Python online submissions for Kth Smallest Element in a Sorted Matrix.

### 原题链接

https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/


您的支持是我最大的动力
