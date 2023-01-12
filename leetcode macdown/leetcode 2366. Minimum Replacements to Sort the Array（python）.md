leetcode  2366. Minimum Replacements to Sort the Array（python）




### 描述

You are given a 0-indexed integer array nums. In one operation you can replace any element of the array with any two elements that sum to it. Return the minimum number of operations to make an array that is sorted in non-decreasing order.



* For example, consider nums = [5,6,7]. In one operation, we can replace nums[1] with 2 and 4 and convert nums to [5,2,4,7].



Example 1:

	Input: nums = [3,9,3]
	Output: 2
	Explanation: Here are the steps to sort the array in non-decreasing order:
	- From [3,9,3], replace the 9 with 3 and 6 so the array becomes [3,3,6,3]
	- From [3,3,6,3], replace the 6 with 3 and 3 so the array becomes [3,3,3,3,3]
	There are 2 steps to sort the array in non-decreasing order. Therefore, we return 2.


	
Example 2:


	Input: nums = [1,2,3,4,5]
	Output: 0
	Explanation: The array is already in non-decreasing order. Therefore, we return 0. 

Example 3:





Note:


	1 <= nums.length <= 10^5
	1 <= nums[i] <= 10^9

### 解析

根据题意，给你一个 0 索引的整数数组 nums。 在一个操作中，您可以将数组中的任何元素替换为任何两个相加的元素。 返回以非递减顺序排序的数组的最小操作数。

* 例如，考虑 nums = [5,6,7]。 在一次操作中，我们可以将 nums[1] 替换为 2 和 4，并将 nums 转换为 [5,2,4,7]。

其实这道题就是考察贪心，我们想要最后的结果是非递减顺序，那就只需要从前往后进行变化，并且使得变化后的最小数字最大即可。最开始我们肯定要使用 nums 的最后一个元素作为基准 mx 来从前往后进行每个元素的操作，有以下不同的情况：

* 如果 nums[-1] <= mx ，那就说明本身符合题目要求，只需要让 mx 更新为 nums[-1] ，去继续当作基准去往前进行操作。
* 如果 nums[-1] > mx ，说明不符合题意，要对 nums[-1] 进行分解，根据题意分解后的值肯定不能超过 mx ，而且要保证分解之后的最小值最大，如果 nums[-1] 能被 mx 整除，那就说明分解之后的最小值依然是 mx ，操作次数 cur // mx - 1 加到结果 result 中。如果 nums[-1] 不能被 mx 整除，那就可以计算出拆分次数为 t = nums[-1] // mx ，加到 result 中即可，同时要使分解之后最小值最大，就要进行均分，所以更新 mx 为 nums //(t+1) 。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。


### 解答

	class Solution(object):
	    def minimumReplacement(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = 0
	        mx = nums[-1]
	        while nums:
	            cur = nums.pop(-1)
	            if cur > mx:
	                if cur % mx == 0:
	                    result += cur // mx - 1
	                else:
	                    c = cur // mx
	                    result += c
	                    mx = cur // (c+1)
	            elif cur <= mx:
	                mx = cur
	        return result

### 运行结果



	47 / 47 test cases passed.
	Status: Accepted
	Runtime: 895 ms
	Memory Usage: 22.9 MB
	
### 原题链接

https://leetcode.com/contest/biweekly-contest-84/problems/minimum-replacements-to-sort-the-array/


您的支持是我最大的动力
