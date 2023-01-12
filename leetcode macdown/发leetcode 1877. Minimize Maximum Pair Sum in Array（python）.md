leetcode  1877. Minimize Maximum Pair Sum in Array（python）

### 描述

The pair sum of a pair (a,b) is equal to a + b. The maximum pair sum is the largest pair sum in a list of pairs.

* For example, if we have pairs (1,5), (2,3), and (4,4), the maximum pair sum would be max(1+5, 2+3, 4+4) = max(6, 5, 8) = 8.

Given an array nums of even length n, pair up the elements of nums into n / 2 pairs such that:

* Each element of nums is in exactly one pair, and
* The maximum pair sum is minimized.
Return the minimized maximum pair sum after optimally pairing up the elements.





Example 1:

	Input: nums = [3,5,2,3]
	Output: 7
	Explanation: The elements can be paired up into pairs (3,3) and (5,2).
	The maximum pair sum is max(3+3, 5+2) = max(6, 7) = 7.

	
Example 2:


	Input: nums = [3,5,4,2,4,6]
	Output: 8
	Explanation: The elements can be paired up into pairs (3,5), (4,4), and (6,2).
	The maximum pair sum is max(3+5, 4+4, 6+2) = max(8, 8, 8) = 8.


Note:

	
	n == nums.length
	2 <= n <= 10^5
	n is even.
	1 <= nums[i] <= 10^5

### 解析


根据题意，给出一个偶数长度的整数列表 nums ，将里面的元素两两组和成一对，每个元素只能属于一个对，使得所有对的和的最大值最小，问这个最小值是多少。细分析一下其实不难，要想使得最大值最小，肯定要让 nums 中最大的元素和最小的元素进行组队求和，次最大的元素和次最小的元素进行组队求和，所以思路很明显就出来了：

* 初始化一个结果变量 result 为 0
* 对 nums 进行升序排序
* 遍历索引 [0, len(nums)//2] ，比较 nums[i]+nums[-1-i] 和 result ，将较大值赋给 result 
* 遍历结束得到的 result 即为答案

### 解答
				

	class Solution(object):
	    def minPairSum(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = 0
	        N = len(nums)
	        nums.sort()
	        j = -1
	        i = 0
	        while i<N//2:
	            result = max(result, nums[i]+nums[j])
	            i+=1
	            j-=1
	        return result
            	      
			
### 运行结果


	Runtime: 1132 ms, faster than 39.81% of Python online submissions for Minimize Maximum Pair Sum in Array.
	Memory Usage: 25.9 MB, less than 37.96% of Python online submissions for Minimize Maximum Pair Sum in Array.
	
原题链接：https://leetcode.com/problems/minimize-maximum-pair-sum-in-array/



您的支持是我最大的动力
