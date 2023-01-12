leetcode  1343. Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold（python）

### 描述

Given an array of integers arr and two integers k and threshold.

Return the number of sub-arrays of size k and average greater than or equal to threshold.





Example 1:

	Input: arr = [2,2,2,2,5,5,5,8], k = 3, threshold = 4
	Output: 3
	Explanation: Sub-arrays [2,5,5],[5,5,5] and [5,5,8] have averages 4, 5 and 6 respectively. All other sub-arrays of size 3 have averages less than 4 (the threshold).

	
Example 2:


	Input: arr = [1,1,1,1,1], k = 1, threshold = 0
	Output: 5

Example 3:


	Input: arr = [11,13,17,23,29,31,7,5,2,3], k = 3, threshold = 5
	Output: 6
	Explanation: The first 6 sub-arrays of size 3 have averages greater than 5. Note that averages are not integers.
	
Example 4:

	Input: arr = [7,7,7,7,7,7,7], k = 7, threshold = 7
	Output: 1

	
Example 5:


	Input: arr = [4,4,4,4], k = 4, threshold = 1
	Output: 1

Note:


	1 <= arr.length <= 10^5
	1 <= arr[i] <= 10^4
	1 <= k <= arr.length
	0 <= threshold <= 10^4

### 解析


根据题意，就是给出了一个整数列表 arr ，然后判断 arr 中的 k 个元素组成的子列表的平均数能够大于等于 threshold 的有多少个，最简单的方法就是暴力解法，直接从左到右遍历 k 元素所能组成的所有子列表，然后用计数器 count 统计平均值大于等于 threshold 的个数即可，但是运行结果是超时，因为题目中 arr 的长度最大可以达到 10^5 。

### 解答
				

	class Solution(object):
	    def numOfSubarrays(self, arr, k, threshold):
	        """
	        :type arr: List[int]
	        :type k: int
	        :type threshold: int
	        :rtype: int
	        """
	        count = 0
	        idx = 0
	        while idx+k<=len(arr):
	            tmp = arr[idx:idx+k]
	            if sum(tmp)/len(tmp) >= threshold:
	                count+=1
	            idx += 1
	        return count
            	      
			
### 运行结果

	Time Limit Exceeded


### 解析


其实主要的耗时操作都在取自列表、求和、除法这三个关键的步骤上面，我们通过使用滑动窗口，来缩短计算时间，也就是每次只计算长度为 k 的窗口大小的和记录为 total ，向右滑动一个元素，计算新的窗口的和就是减去之前窗口的第一个元素且加上新进入窗口的最后一个元素，这样就可以大幅度减少求和的运算。


### 解答
				

	class Solution(object):
	    def numOfSubarrays(self, arr, k, threshold):
	        """
	        :type arr: List[int]
	        :type k: int
	        :type threshold: int
	        :rtype: int
	        """
	        count = 0
	        threshold = k*threshold
	        total = sum(arr[:k])
	        if total>=threshold:
	            count += 1
	        for i in range(k,len(arr)):
	            total = total+arr[i]-arr[i-k]
	            if total >= threshold:
	                count += 1
	        return count
            	      
			
### 运行结果

	Runtime: 488 ms, faster than 100.00% of Python online submissions for Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold.
	Memory Usage: 23.2 MB, less than 100.00% of Python online submissions for Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold.

原题链接：https://leetcode.com/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/



您的支持是我最大的动力
