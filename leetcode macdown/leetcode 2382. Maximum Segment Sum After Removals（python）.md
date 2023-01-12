leetcode  2382. Maximum Segment Sum After Removals（python）




### 描述

You are given two 0-indexed integer arrays nums and removeQueries, both of length n. For the ith query, the element in nums at the index removeQueries[i] is removed, splitting nums into different segments. A segment is a contiguous sequence of positive integers in nums. A segment sum is the sum of every element in a segment.

Return an integer array answer, of length n, where answer[i] is the maximum segment sum after applying the ith removal. Note: The same index will not be removed more than once.



Example 1:


	Input: nums = [1,2,5,6,1], removeQueries = [0,3,2,4,1]
	Output: [14,7,2,2,0]
	Explanation: Using 0 to indicate a removed element, the answer is as follows:
	Query 1: Remove the 0th element, nums becomes [0,2,5,6,1] and the maximum segment sum is 14 for segment [2,5,6,1].
	Query 2: Remove the 3rd element, nums becomes [0,2,5,0,1] and the maximum segment sum is 7 for segment [2,5].
	Query 3: Remove the 2nd element, nums becomes [0,2,0,0,1] and the maximum segment sum is 2 for segment [2]. 
	Query 4: Remove the 4th element, nums becomes [0,2,0,0,0] and the maximum segment sum is 2 for segment [2]. 
	Query 5: Remove the 1st element, nums becomes [0,0,0,0,0] and the maximum segment sum is 0, since there are no segments.
	Finally, we return [14,7,2,2,0].
	
Example 2:


	Input: nums = [3,2,11,1], removeQueries = [3,2,1,0]
	Output: [16,5,3,0]
	Explanation: Using 0 to indicate a removed element, the answer is as follows:
	Query 1: Remove the 3rd element, nums becomes [3,2,11,0] and the maximum segment sum is 16 for segment [3,2,11].
	Query 2: Remove the 2nd element, nums becomes [3,2,0,0] and the maximum segment sum is 5 for segment [3,2].
	Query 3: Remove the 1st element, nums becomes [3,0,0,0] and the maximum segment sum is 3 for segment [3].
	Query 4: Remove the 0th element, nums becomes [0,0,0,0] and the maximum segment sum is 0, since there are no segments.
	Finally, we return [16,5,3,0].



Note:


	n == nums.length == removeQueries.length
	1 <= n <= 10^5
	1 <= nums[i] <= 10^9
	0 <= removeQueries[i] < n
	All the values of removeQueries are unique.

### 解析

根据题意，给定两个长度为 n 的 0 索引整数数组 nums 和 removeQueries。遍历 removeQueries ，对于每个 removeQueries[i]  表示要执行删除 nums 中的索引为 removeQueries[i] 的元素的操作，将 nums 分成不同的段。 段是一个连续正整数序列。 段总和是段中每个元素的总和。返回一个长度为 n 的整数数组 answer ，其中 answer[i] 是执行了第 i 次删除操作后所能出现的最大段总和。

模拟题意正向删除元素的做法不容易实现，我们进行逆向操作，想象一下我们有一个空数组 newNum ，然后不断往里按照逆序的 removeQueries 对应的 nums[i] 进行添加元素的操作，在添加元素的过程中，如果其左/右边界已经出现了相邻的段，则对左/右边相邻段进行合并，并更新段的左右边界和当前段的和，并且更新当前出现的最大的段的和加入到 result 中。需要注意的是我们只需要逆向遍历到 removeQueries 的第二个元素为止即可，因为将所有元素都删除的最大段和肯定是 0 ，我们在一开始就加入到 result 中了，最后将 result[::-1] 返回即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def maximumSegmentSum(self, nums, removeQueries):
	        """
	        :type nums: List[int]
	        :type removeQueries: List[int]
	        :rtype: List[int]
	        """
	        L_Merge, R_Merge = {}, {}
	        result = [0]
	        mx = 0
	        for i in removeQueries[::-1][:-1]:
	            tmpSum = nums[i]
	            leftB = rightB = i
	            if i + 1 in L_Merge:
	                tmpSum += L_Merge[i + 1][0]
	                rightB = L_Merge[i + 1][2]
	            if i - 1 in R_Merge:
	                tmpSum += R_Merge[i - 1][0]
	                leftB = R_Merge[i - 1][1]
	            L_Merge[leftB] = (tmpSum, leftB, rightB)
	            R_Merge[rightB] = (tmpSum, leftB, rightB)
	            mx = max(mx, tmpSum)
	            result.append(mx)
	        return result[::-1]
	

	


### 运行结果

	Runtime: 1372 ms, faster than 100.00% of Python online submissions for Maximum Segment Sum After Removals.
	Memory Usage: 43.9 MB, less than 100.00% of Python online submissions for Maximum Segment Sum After Removals.


### 原题链接


https://leetcode.com/contest/biweekly-contest-85/problems/maximum-segment-sum-after-removals/

您的支持是我最大的动力
