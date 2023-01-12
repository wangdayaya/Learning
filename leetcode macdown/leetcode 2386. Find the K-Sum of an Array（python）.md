leetcode  2386. Find the K-Sum of an Array（python）




### 描述


You are given an integer array nums and a positive integer k. You can choose any subsequence of the array and sum all of its elements together. We define the K-Sum of the array as the k<sup>th</sup> largest subsequence sum that can be obtained (not necessarily distinct). Return the K-Sum of the array.

A subsequence is an array that can be derived from another array by deleting some or no elements without changing the order of the remaining elements.Note that the empty subsequence is considered to have a sum of 0.


Example 1:

	Input: nums = [2,4,-2], k = 5
	Output: 2
	Explanation: All the possible subsequence sums that we can obtain are the following sorted in decreasing order:
	- 6, 4, 4, 2, 2, 0, 0, -2.
	The 5-Sum of the array is 2.

	
Example 2:


	Input: nums = [1,-2,3,4,-10,12], k = 16
	Output: 10
	Explanation: The 16-Sum of the array is 10.

Note:

	n == nums.length
	1 <= n <= 10^5
	-10^9 <= nums[i] <= 10^9
	1 <= k <= min(2000, 2n^)


### 解析

根据题意，给定一个整数数组 nums 和一个正整数 k 。 我们可以选择数组的任何子序列并将其所有元素相加。我们将数组的 K-Sum 定义为可以获得的第 k 个最大子序列和（不一定是不同的）。返回数组的 K-Sum。子序列是一个数组，可以通过删除一些元素或不删除元素而不改变剩余元素的顺序从另一个数组派生。 请注意，空子序列的总和为 0 。

其实我们很容易就想到所有可能序列的最大和肯定是所有正整数的和 mxSum ，此时我们可以通过不断减小 mxSum 来取得可能的其他子序列的和，可能是加一个负数，也可能是减一个正数，但是对原数组 nums 都进行绝对值化之后得到 absNum ，我们发现其实本质上这个操作就都变成了减一个正数，这样不断减小 mxSum ，就能得到第 k 个大的序列和。

解决这种问题最合适的数据结构就是堆 maxHeap ，每个堆的对象中存放的就是 （当前最大子序列和 curMnSum ，absNum 中下一个大的元素的索引）。我们对 absNum 先进行排序，当我们弹出 curMnSum ，将其加入到小根堆 result 中，我们找下一个可能的最小子序列会有两种可能方式出现：
	
* 	将 absNum[i]  替换为 absNum[i+1]  
* 	直接将下一个 absNum[i+1]  到加入 curMnSum 


这两种情况都有可能生成下一个最小的非空子序列和，所以都要压入堆中，最后得到长度为 k 的 result 之后返回 result[0] 就是最后的答案。

时间复杂度为 O(NlogN + klogk) ，空间复杂度为 O(N+k) 。





### 解答

	class Solution(object):
	    def kSum(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: int
	        """
	        mxSum = sum([num for num in nums if num>=0])
	        absNum = sorted([abs(num) for num in nums])
	        maxHeap = [(-mxSum + absNum[0], 0)]
	        result = [mxSum]
	        while len(result) < k:
	            nextSum, i = heapq.heappop(maxHeap)
	            heapq.heappush(result, -nextSum)
	            if i + 1 < len(absNum):
	                heapq.heappush(maxHeap, (nextSum - absNum[i] + absNum[i + 1], i + 1))
	                heapq.heappush(maxHeap, (nextSum + absNum[i + 1], i + 1))
	        return result[0]

### 运行结果

	
	111 / 111 test cases passed.
	Status: Accepted
	Runtime: 1134 ms
	Memory Usage: 25.7 MB

### 原题链接


https://leetcode.com/contest/weekly-contest-307/problems/find-the-k-sum-of-an-array/

您的支持是我最大的动力
