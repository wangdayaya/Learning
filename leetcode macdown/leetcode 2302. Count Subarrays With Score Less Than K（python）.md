leetcode 2302. Count Subarrays With Score Less Than K （python）




### 描述


The score of an array is defined as the product of its sum and its length.

* For example, the score of [1, 2, 3, 4, 5] is (1 + 2 + 3 + 4 + 5) * 5 = 75.

Given a positive integer array nums and an integer k, return the number of non-empty subarrays of nums whose score is strictly less than k. A subarray is a contiguous sequence of elements within an array.


Example 1:

	Input: nums = [2,1,4,3,5], k = 10
	Output: 6
	Explanation:
	The 6 subarrays having scores less than 10 are:
	- [2] with score 2 * 1 = 2.
	- [1] with score 1 * 1 = 1.
	- [4] with score 4 * 1 = 4.
	- [3] with score 3 * 1 = 3. 
	- [5] with score 5 * 1 = 5.
	- [2,1] with score (2 + 1) * 2 = 6.
	Note that subarrays such as [1,4] and [4,3,5] are not considered because their scores are 10 and 36 respectively, while we need scores strictly less than 10.

	
Example 2:


	Input: nums = [1,1,1], k = 5
	Output: 5
	Explanation:
	Every subarray except [1,1,1] has a score less than 5.
	[1,1,1] has a score (1 + 1 + 1) * 3 = 9, which is greater than 5.
	Thus, there are 5 subarrays having scores less than 5.




Note:


	1 <= nums.length <= 10^5
	1 <= nums[i] <= 10^5
	1 <= k <= 10^15

### 解析

根据题意，数组的分数定义为其总和与其长度的乘积。

* 例如 [1, 2, 3, 4, 5] 的得分为 (1 + 2 + 3 + 4 + 5) * 5 = 75。

给定一个正整数数组 nums 和一个整数 k ，返回分数小于 k 的 nums 的非空子数组的数量。子数组是数组中元素的连续序列。

这道题考查的就是滑动窗口常规解题方法，我们要找一个子数组分数小于 k 的数量，所以我们只要定义一个滑动窗口列表 window ，然后遍历数组 nums 中的每个元素 n ，将其加入 window ，如果 window 当前的分数大于等于 k 则不符合题意，那么我们就将其最左边的元素不断弹出，直到满足小于 k 的要求，然后将当前的 window 长度加入 result 中，遍历下一个元素不断重复上述操作，最后得到的 result 即为结果。

时间复杂度为 (N) ，空间复杂度为 O(N) 。

### 解答
				

	class Solution(object):
	    def countSubarrays(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: int
	        """
	        result = 0
	        total = 0
	        window = []
	        for n in nums:
	            total += n
	            window.append(n)
	            while window and total * len(window) >= k:
	                total -= window.pop(0)
	            result += len(window)
	        return result
            	      
			
### 运行结果

	167 / 167 test cases passed.
	Status: Accepted
	Runtime: 2657 ms
	Memory Usage: 24.2 MB

### 解析

其实上面的解法本质上也就是双指针解法，只不过我们具像化为了一个滑动窗口列表，我们可以使用 L 和 R 的指针来完成解题。我们遍历的每个 nums 中的元素 n ，索引为 R ，不断将元素累加到 sum 中，如果 sum \* (R-L+1) 大于等于 k ，说明以 L 为左边界不合适，需要不断将 sum 减去最左边的值，同时 L 加一 ，然后将 R-L+1 加入 result 表示在索引 R 为右边界的时候，有多少个左边界所构成的子数组都符合题意。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。


### 解答

	class Solution(object):
	    def countSubarrays(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: int
	        """
	        result = 0
	        L = sum = 0
	        for R,n in enumerate(nums):
	            sum += n
	            while sum * (R-L+1) >= k:
	                sum -= nums[L]
	                L += 1
	            result += R-L+1
	        return result
	        
### 运行结果

	167 / 167 test cases passed.
	Status: Accepted
	Runtime: 1740 ms
	Memory Usage: 24.2 MB

### 原题链接



https://leetcode.com/contest/biweekly-contest-80/problems/count-subarrays-with-score-less-than-k/


您的支持是我最大的动力
