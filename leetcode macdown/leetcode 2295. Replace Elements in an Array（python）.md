leetcode 2295. Replace Elements in an Array （python）



### 描述


You are given a 0-indexed array nums that consists of n distinct positive integers. Apply m operations to this array, where in the ith operation you replace the number operations[i][0] with operations[i][1].

It is guaranteed that in the ith operation:

operations[i][0] exists in nums.
operations[i][1] does not exist in nums.
Return the array obtained after applying all the operations.


Example 1:

	Input: nums = [1,2,4,6], operations = [[1,3],[4,7],[6,1]]
	Output: [3,2,7,1]
	Explanation: We perform the following operations on nums:
	- Replace the number 1 with 3. nums becomes [3,2,4,6].
	- Replace the number 4 with 7. nums becomes [3,2,7,6].
	- Replace the number 6 with 1. nums becomes [3,2,7,1].
	We return the final array [3,2,7,1].

	
Example 2:


	Input: nums = [1,2], operations = [[1,3],[2,1],[3,2]]
	Output: [2,1]
	Explanation: We perform the following operations to nums:
	- Replace the number 1 with 3. nums becomes [3,2].
	- Replace the number 2 with 1. nums becomes [3,1].
	- Replace the number 3 with 2. nums becomes [2,1].
	We return the array [2,1].






Note:

	n == nums.length
	m == operations.length
	1 <= n, m <= 10^5
	All the values of nums are distinct.
	operations[i].length == 2
	1 <= nums[i], operations[i][0], operations[i][1] <= 10^6
	operations[i][0] will exist in nums when applying the ith operation.
	operations[i][1] will not exist in nums when applying the ith operation.


### 解析

根据题意，给定一个由 n 个不同的正整数组成的索引为 0 的数组 nums 。 对这个数组应用 m 个操作，在第 i 个操作中，将数字  operations[i][0] 替换为 operations[i][1] 。

题目保证在第 i 次操作中：

* operation[i][0] 存在于 nums 中 
* operation[i][1] 在 nums 中不存在 

返回应用所有操作后得到的数组。


这道题其实按照题意就可以了，需要注意的地方是我们要使用字典 d 来保存每个元素的索引，并且在进行操作之后将每个字符对应的索引进行更新即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答
				
		class Solution(object):
		    def arrayChange(self, nums, operations):
		        """
		        :type nums: List[int]
		        :type operations: List[List[int]]
		        :rtype: List[int]
		        """
		        N = len(operations)
		        d = {}
		        for i, c in enumerate(nums):
		            d[c] = i
		        for i in range(N):
		            x, y = operations[i]
		            idx = d[x]
		            nums[idx] = y
		            d[y] = idx
		        return nums

            	      
			
### 运行结果


	80 / 80 test cases passed.
	Status: Accepted
	Runtime: 1826 ms
	Memory Usage: 73.6 MB
        


### 原题链接

https://leetcode.com/contest/weekly-contest-296/problems/replace-elements-in-an-array/

您的支持是我最大的动力
