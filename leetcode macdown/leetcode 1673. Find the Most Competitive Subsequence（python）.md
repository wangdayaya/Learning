leetcode  1673. Find the Most Competitive Subsequence（python）

### 每日经典

《早发白帝城》 ——李白（唐）


朝辞白帝彩云间，千里江陵一日还。
 
两岸猿声啼不住，轻舟已过万重山。
### 描述



Given an integer array nums and a positive integer k, return the most competitive subsequence of nums of size k.

An array's subsequence is a resulting sequence obtained by erasing some (possibly zero) elements from the array.

We define that a subsequence a is more competitive than a subsequence b (of the same length) if in the first position where a and b differ, subsequence a has a number less than the corresponding number in b. For example, [1,3,4] is more competitive than [1,3,5] because the first position they differ is at the final number, and 4 is less than 5.

Example 1:

	Input: nums = [3,5,2,6], k = 2
	Output: [2,6]
	Explanation: Among the set of every possible subsequence: {[3,5], [3,2], [3,6], [5,2], [5,6], [2,6]}, [2,6] is the most competitive.

	
Example 2:

	Input: nums = [2,4,3,3,5,4,9,6], k = 4
	Output: [2,3,3,4]





Note:

	1 <= nums.length <= 10^5
	0 <= nums[i] <= 10^9
	1 <= k <= nums.length


### 解析


根据题意，给定一个整数数组 nums 和一个正整数 k，返回大小为 k 的 nums 中最具竞争力的子序列。题目定义，如果在 a 和 b 第一个数字不同的位置，子序列 a 的数字小于 b 中的相应数字，则子序列 a 比子序列 b（相同长度）更具竞争力。 例如，[1,3,4] 比 [1,3,5] 更具竞争力，因为 4 小于 5。

这道题很明显还是考察单调栈的使用和贪心思想，要想找最具竞争力的子序列，所以维护一个单调栈 stack ，把尽可能小的数字尽可能放到前面，但是这里要注意最后要得到一个 k 长度的子序列，所以要始终保证 stack 中序列的长度加上剩下的所有可用数字总和至少有 k 个，也就是只能最多浪费 len(nums)-k 个数字。

### 解答
				
	class Solution(object):
	    def mostCompetitive(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: List[int]
	        """
	        available = len(nums)-k
	        stack = []
	        for n in nums:
	            while stack and available>0 and stack[-1]>n :
	                stack.pop()
	                available -= 1
	            stack.append(n)
	        return stack[:k]

            	      
			
### 运行结果
	
	Runtime: 1795 ms, faster than 10.34% of Python online submissions for Find the Most Competitive Subsequence.
	Memory Usage: 26.4 MB, less than 96.55% of Python online submissions for Find the Most Competitive Subsequence.


原题链接：https://leetcode.com/problems/find-the-most-competitive-subsequence/



您的支持是我最大的动力
