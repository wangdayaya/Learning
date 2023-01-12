leetcode  2281. Sum of Total Strength of Wizards（python）



### 描述

As the ruler of a kingdom, you have an army of wizards at your command.

You are given a 0-indexed integer array strength, where strength[i] denotes the strength of the ith wizard. For a contiguous group of wizards (i.e. the wizards' strengths form a subarray of strength), the total strength is defined as the product of the following two values:

* The strength of the weakest wizard in the group.
* The total of all the individual strengths of the wizards in the group.

Return the sum of the total strengths of all contiguous groups of wizards. Since the answer may be very large, return it modulo 109 + 7.

A subarray is a contiguous non-empty sequence of elements within an array.



Example 1:


	Input: strength = [1,3,1,2]
	Output: 44
	Explanation: The following are all the contiguous groups of wizards:
	- [1] from [1,3,1,2] has a total strength of min([1]) * sum([1]) = 1 * 1 = 1
	- [3] from [1,3,1,2] has a total strength of min([3]) * sum([3]) = 3 * 3 = 9
	- [1] from [1,3,1,2] has a total strength of min([1]) * sum([1]) = 1 * 1 = 1
	- [2] from [1,3,1,2] has a total strength of min([2]) * sum([2]) = 2 * 2 = 4
	- [1,3] from [1,3,1,2] has a total strength of min([1,3]) * sum([1,3]) = 1 * 4 = 4
	- [3,1] from [1,3,1,2] has a total strength of min([3,1]) * sum([3,1]) = 1 * 4 = 4
	- [1,2] from [1,3,1,2] has a total strength of min([1,2]) * sum([1,2]) = 1 * 3 = 3
	- [1,3,1] from [1,3,1,2] has a total strength of min([1,3,1]) * sum([1,3,1]) = 1 * 5 = 5
	- [3,1,2] from [1,3,1,2] has a total strength of min([3,1,2]) * sum([3,1,2]) = 1 * 6 = 6
	- [1,3,1,2] from [1,3,1,2] has a total strength of min([1,3,1,2]) * sum([1,3,1,2]) = 1 * 7 = 7
	The sum of all the total strengths is 1 + 9 + 1 + 4 + 4 + 4 + 3 + 5 + 6 + 7 = 44.
	
Example 2:


	Input: strength = [5,4,6]
	Output: 213
	Explanation: The following are all the contiguous groups of wizards: 
	- [5] from [5,4,6] has a total strength of min([5]) * sum([5]) = 5 * 5 = 25
	- [4] from [5,4,6] has a total strength of min([4]) * sum([4]) = 4 * 4 = 16
	- [6] from [5,4,6] has a total strength of min([6]) * sum([6]) = 6 * 6 = 36
	- [5,4] from [5,4,6] has a total strength of min([5,4]) * sum([5,4]) = 4 * 9 = 36
	- [4,6] from [5,4,6] has a total strength of min([4,6]) * sum([4,6]) = 4 * 10 = 40
	- [5,4,6] from [5,4,6] has a total strength of min([5,4,6]) * sum([5,4,6]) = 4 * 15 = 60
	The sum of all the total strengths is 25 + 16 + 36 + 36 + 40 + 60 = 213.


Note:

	1 <= strength.length <= 10^5
	1 <= strength[i] <= 10^9


### 解析


根据题意，王国的统治者有一支可以指挥的巫师大军。给定一个索引为 0 的整数数组 strength ，其中 strength[i] 表示第 i 个方向的力量。 对于一组连续的巫师，总力量定义为以下两个值的乘积：

* 团体中最弱的巫师的实力
* 团体巫师所有个人力量的总和

返回所有相邻巫师组的总力量之和。 由于答案可能非常大，因此以 10^9 + 7 为模返回。

我们只需要通过单调栈找出每个元素左边比它小的元素的索引 L ，以及右边比它小的索引 R ，这样我们就计算每个元素然后在这个范围 [L,R] 内各个子数组的巫师力量，因为数据量比较大，双重循环计算会超时，所以我们经过数学公式的推导使用前缀和的前缀和来减少运算量，具体推演过程比较复杂，可以参考[灵神的解答](https://leetcode.cn/problems/sum-of-total-strength-of-wizards/solution/dan-diao-zhan-qian-zhui-he-de-qian-zhui-d9nki/)。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答
				
	
	class Solution:
	    def totalStrength(self, S: List[int]) -> int:
	        N = len(S)
	        left = [-1] * N
	        stack = []
	        for i,v in enumerate(S):
	            while stack and S[stack[-1]] >= v:
	                stack.pop()
	            if stack:
	                left[i] = stack[-1]
	            stack.append(i)
	
	        right = [N] * N
	        stack = []
	        for i in range(N-1, -1, -1):
	            v = S[i]
	            while stack and S[stack[-1]] > v:
	                stack.pop()
	            if stack:
	                right[i] = stack[-1]
	            stack.append(i)
	            
	        SS = [0]+list(accumulate([0]+list(accumulate(S))))
	        result = 0
	        for i,v in enumerate(S):
	            L,R = left[i]+1, right[i]-1
	            total = (i-L+1)*(SS[R+2] - SS[i+1]) - (R-i+1) * (SS[i+1] - SS[L])
	            result += total * v
	        return result % (10**9+7)
	

            	      
			
### 运行结果

	Runtime: 1601 ms, faster than 90.22% of Python3 online submissions for Sum of Total Strength of Wizards.
	Memory Usage: 37.5 MB, less than 40.04% of Python3 online submissions for Sum of Total Strength of Wizards.


### 原题链接

https://leetcode.com/contest/weekly-contest-294/problems/sum-of-total-strength-of-wizards/


您的支持是我最大的动力
