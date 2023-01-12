leetcode 74. Search a 2D Matrix （python）




### 描述

Write an efficient algorithm that searches for a value target in an m x n integer matrix matrix. This matrix has the following properties:

* Integers in each row are sorted from left to right.
* The first integer of each row is greater than the last integer of the previous row.




Example 1:

![](https://assets.leetcode.com/uploads/2020/10/05/mat.jpg)

	Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
	Output: true

	
Example 2:


![](https://assets.leetcode.com/uploads/2020/10/05/mat2.jpg)

	Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
	Output: false






Note:

	m == matrix.length
	n == matrix[i].length
	1 <= m, n <= 100
	-10^4 <= matrix[i][j], target <= 10^4



### 解析
根据题意，编写一个在 m x n 整数矩阵 matrix 中搜索值 target 的高效算法。 该矩阵具有以下性质：

* 每行中的整数从左到右升序排序
* 每行的第一个整数大于前一行的最后一个整数

其实这道题如果相解决是很简单啊的，就是把所有的元素都遍历一次就可以了，这样做的时间复杂度为 O(N) ，空间复杂度为 O(1) 。但是这种解法太简单粗暴了，如果 matrix 的尺寸是非常大的矩阵，例如 10^8 左右，那么这种就会超时了，肯定要寻求更加节省时间的算法。

这道题其实已经给出的信号很明显了，每行从左到右升序，下一行的数都比上一行的大，所以这种肯定是要用二分查找分进行 target 的搜索。思路比较简单：

* 首先我们遍历每一行数据 matrix[i] ，如果 matrix[i][0] 或者 matrix[i][-1] 等于 target ，那么就直接返回 True ； 
* 否则判断 target 是不是在 matrix[i] 行数据范围之内，如果在的话进行二分查找即可；如果不在进行下一行的判断；
* 如果遍历结束还没有结果直接返回 False

如果比较这两种算法，很明显使用了二分搜索的方法更加省时，时间复杂度为 O(logN) ，空间复杂度为 O(1) 。


### 解答
				

	class Solution(object):
	    def searchMatrix(self, matrix, target):
	        """
	        :type matrix: List[List[int]]
	        :type target: int
	        :rtype: bool
	        """
	        for i in range(len(matrix)):
	            if target == matrix[i][0] or target == matrix[i][-1]:
	                return True
	            if matrix[i][0]  < target < matrix[i][-1] :
	                return self.search(matrix[i], target)
	        return False
	    
	    def search(self, nums, target):
	        i = 0
	        j = len(nums) - 1
	        while i <= j:
	            mid = (i+j) // 2
	            if nums[mid] == target:
	                return True
	            elif nums[mid] > target:
	                j = mid - 1
	            else:
	                i = mid + 1
	        return False
        
            	      
			
### 运行结果


	Runtime: 28 ms, faster than 92.46% of Python online submissions for Search a 2D Matrix.
	Memory Usage: 13.9 MB, less than 19.92% of Python online submissions for Search a 2D Matrix.

### 原题链接


https://leetcode.com/problems/search-a-2d-matrix/


您的支持是我最大的动力
