leetcode  456. 132 Pattern（python）




### 描述


Given an array of n integers nums, a 132 pattern is a subsequence of three integers nums[i], nums[j] and nums[k] such that i < j < k and nums[i] < nums[k] < nums[j].

Return true if there is a 132 pattern in nums, otherwise, return false.




Example 1:


	Input: nums = [1,2,3,4]
	Output: false
	Explanation: There is no 132 pattern in the sequence.
	
Example 2:

	Input: nums = [3,1,4,2]
	Output: true
	Explanation: There is a 132 pattern in the sequence: [1, 4, 2].


Example 3:


	Input: nums = [-1,3,2,0]
	Output: true
	Explanation: There are three 132 patterns in the sequence: [-1, 3, 2], [-1, 3, 0] and [-1, 2, 0].
	


Note:

	n == nums.length
	1 <= n <= 2 * 10^5
	-10^9 <= nums[i] <= 10^9


### 解析


根据题意，给定一个包含 n 个整数 nums 的数组，132 模式是三个整数 nums[i]、nums[j] 和 nums[k] 的子序列，使得 i < j < k 和 nums[i] < nums[k] < nums [j]。

如果 nums 中有 132 个模式，则返回 true，否则返回 false。

### 解答
				


            	      
			
### 运行结果




### 原题链接



https://leetcode.com/problems/132-pattern/


您的支持是我最大的动力
