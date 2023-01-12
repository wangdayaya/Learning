leetcode  1200. Minimum Absolute Difference（python）

### 描述


Given an array of distinct integers arr, find all pairs of elements with the minimum absolute difference of any two elements. 

Return a list of pairs in ascending order(with respect to pairs), each pair [a, b] follows:

* a, b are from arr
* a < b
* b - a equals to the minimum absolute difference of any two elements in arr


Example 1:

	Input: arr = [4,2,1,3]
	Output: [[1,2],[2,3],[3,4]]
	Explanation: The minimum absolute difference is 1. List all pairs with difference equal to 1 in ascending order.

	
Example 2:

	Input: arr = [1,3,6,10,15]
	Output: [[1,3]]


Example 3:

	Input: arr = [3,8,-10,23,19,-4,-14,27]
	Output: [[-14,-10],[19,23],[23,27]]



Note:
	
	2 <= arr.length <= 10^5
	-10^6 <= arr[i] <= 10^6



### 解析


根据题意，就是找出 arr 中差值最小的数值对有哪些。直接遍历经过排序的 arr ，如果两个相邻的元素差值比 min_diatance 小，那就更新 min_diatance ，然后更新 result 为空列表，并将这两个相邻的数值对放进去，如果两个相邻的元素差值等于 min_diatance ，将这两个相邻的数值对追加到 result 中，遍历 arr 结束之后，即可得到结果 。

### 解答
				

	class Solution(object):
	    def minimumAbsDifference(self, arr):
	        """
	        :type arr: List[int]
	        :rtype: List[List[int]]
	        """
	        arr.sort()
	        i = 1
	        min_diatance = float("inf")
	        result = []
	        while i < len(arr):
	            distance = arr[i] - arr[i - 1]
	            if distance < min_diatance:
	                min_diatance =  distance
	                result = [[arr[i - 1], arr[i]]]
	            elif distance == min_diatance:
	                result.append([arr[i - 1], arr[i]])
	            i += 1
	        return result
            	      
			
### 运行结果


	Runtime: 304 ms, faster than 73.97% of Python online submissions for Minimum Absolute Difference.
	Memory Usage: 24.4 MB, less than 42.56% of Python online submissions for Minimum Absolute Difference.

### 解析

另外,可以用比较高级的 python 内置函数 zip 来解决这个问题，先将 arr 从小到大进行排序，然后使用 zip 将 arr 和 arr[1:] 进行两两的差值计算，并得到最小的差值 m ，然后再重新遍历  arr 和 arr[1:] 组合而成的 zip 链，将差值为 m 的数值对找出来放入列表中即可。

这里简单介绍下 zip 的功能。zip 函数可以将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。举例：

	a = [1,2,3]
	b = [4,5,6]
	c = [4,5,6,7,8]
	# 打包为元组的列表
	print([x for x in zip(a,b)])
	# 元素个数与最短的列表一致
	print([x for x in zip(a,c)])
	# # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
	print([x for x in zip(*zip(a,b))])

打印：

	[(1, 4), (2, 5), (3, 6)]
	[(1, 4), (2, 5), (3, 6)]
	[(1, 2, 3), (4, 5, 6)]
### 解答
				
	class Solution(object):
	    def minimumAbsDifference(self, arr):
	        """
	        :type arr: List[int]
	        :rtype: List[List[int]]
	        """
	        arr.sort()
	        m = min(j-i for i,j in zip(arr,arr[1:]))
	        return [[i,j] for i,j in zip(arr,arr[1:]) if j-i==m]

            	      
			
### 运行结果

	Runtime: 340 ms, faster than 44.24% of Python online submissions for Minimum Absolute Difference.
	Memory Usage: 32.2 MB, less than 14.55% of Python online submissions for Minimum Absolute Difference.

原题链接：https://leetcode.com/problems/minimum-absolute-difference/



您的支持是我最大的动力
