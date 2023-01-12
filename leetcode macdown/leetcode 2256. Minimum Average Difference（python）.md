leetcode  2256. Minimum Average Difference（python）


这是第 77 场双周赛的第二题，难度 Medium ，考察的是字符串的基本操作。


### 描述


You are given a 0-indexed integer array nums of length n.

The average difference of the index i is the absolute difference between the average of the first i + 1 elements of nums and the average of the last n - i - 1 elements. Both averages should be rounded down to the nearest integer.

Return the index with the minimum average difference. If there are multiple such indices, return the smallest one.

Note:

* The absolute difference of two numbers is the absolute value of their difference.
* The average of n elements is the sum of the n elements divided (integer division) by n.
* The average of 0 elements is considered to be 0.


Example 1:

	Input: nums = [2,5,3,9,5,3]
	Output: 3
	Explanation:
	- The average difference of index 0 is: |2 / 1 - (5 + 3 + 9 + 5 + 3) / 5| = |2 / 1 - 25 / 5| = |2 - 5| = 3.
	- The average difference of index 1 is: |(2 + 5) / 2 - (3 + 9 + 5 + 3) / 4| = |7 / 2 - 20 / 4| = |3 - 5| = 2.
	- The average difference of index 2 is: |(2 + 5 + 3) / 3 - (9 + 5 + 3) / 3| = |10 / 3 - 17 / 3| = |3 - 5| = 2.
	- The average difference of index 3 is: |(2 + 5 + 3 + 9) / 4 - (5 + 3) / 2| = |19 / 4 - 8 / 2| = |4 - 4| = 0.
	- The average difference of index 4 is: |(2 + 5 + 3 + 9 + 5) / 5 - 3 / 1| = |24 / 5 - 3 / 1| = |4 - 3| = 1.
	- The average difference of index 5 is: |(2 + 5 + 3 + 9 + 5 + 3) / 6 - 0| = |27 / 6 - 0| = |4 - 0| = 4.
	The average difference of index 3 is the minimum average difference so return 3.

	
Example 2:

	Input: nums = [0]
	Output: 0
	Explanation:
	The only index is 0 so return 0.
	The average difference of index 0 is: |0 / 1 - 0| = |0 - 0| = 0.






Note:

1 <= nums.length <= 10^5
0 <= nums[i] <= 10^5


### 解析


根据题意，给定一个长度为 n 的 0 索引整数数组 nums 。索引 i 的平均差定义为 nums 的前 i + 1 个元素的平均值与最后 n - i - 1 个元素的平均值之间的绝对差。 两个平均值都应向下舍入到最接近的整数。返回具有最小平均差的索引。如果有多个这样的索引，则返回最小的一个。

这道题很明显考察的就是前缀和，因为我们要求平均差，就要先求出前 n 个元素的和以及后面 m 个元素的和，这样我们就需要维护两个累加和列表 prev 和 back ，prev 就是从前到后的每个位置的累加和，back 就是从后到前每个位置的累积和，然后我们再遍历每个位置 i 的时候，只需要使用 prev[i] 和 back[N - i - 2] 以及各自的个数来计算平均差即可，如果有较小平均差值就使用 result 记录下该索引，遍历结束返回 result 即可。这里需要注意的是当索引到最后一个位置的时候因为有半部分不存在，所以直接计算 prev[i]//N 比较平均差值即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答
				
	class Solution(object):
	    def minimumAverageDifference(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        prev = [nums[0]]
	        back = [nums[-1]]
	        N = len(nums)
	        for i in range(1, N):
	            prev.append(prev[-1]+nums[i])
	            back.append(back[-1]+nums[N-i-1])
	        mn = float('inf')
	        result = 0
	        for i in range(N):
	            if i!=N-1:
	                tmp = abs(prev[i] // (i + 1) - back[N - i - 2] // (N - i - 1))
	                if tmp < mn:
	                    result = i
	                    mn = tmp
	            else:
	                if prev[i]//N < mn:
	                    result = i
	                    mn = prev[i]//N
	        return result

            	      
			
### 运行结果

	78 / 78 test cases passed.
	Status: Accepted
	Runtime: 1010 ms
	Memory Usage: 30 MB


### 原题链接



https://leetcode.com/contest/biweekly-contest-77/problems/minimum-average-difference/


您的支持是我最大的动力
