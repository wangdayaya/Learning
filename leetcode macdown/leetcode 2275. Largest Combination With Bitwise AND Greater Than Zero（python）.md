leetcode 2275. Largest Combination With Bitwise AND Greater Than Zero （python）




### 描述

The bitwise AND of an array nums is the bitwise AND of all integers in nums.

* For example, for nums = [1, 5, 3], the bitwise AND is equal to 1 & 5 & 3 = 1.
* Also, for nums = [7], the bitwise AND is 7.

You are given an array of positive integers candidates. Evaluate the bitwise AND of every combination of numbers of candidates. Each number in candidates may only be used once in each combination.

Return the size of the largest combination of candidates with a bitwise AND greater than 0.



Example 1:

	Input: candidates = [16,17,71,62,12,24,14]
	Output: 4
	Explanation: The combination [16,17,62,24] has a bitwise AND of 16 & 17 & 62 & 24 = 16 > 0.
	The size of the combination is 4.
	It can be shown that no combination with a size greater than 4 has a bitwise AND greater than 0.
	Note that more than one combination may have the largest size.
	For example, the combination [62,12,24,14] has a bitwise AND of 62 & 12 & 24 & 14 = 8 > 0.

	
Example 2:

	Input: candidates = [8,8]
	Output: 2
	Explanation: The largest combination [8,8] has a bitwise AND of 8 & 8 = 8 > 0.
	The size of the combination is 2, so we return 2.






Note:


	1 <= candidates.length <= 10^5
	1 <= candidates[i] <= 10^7

### 解析

根据题意，数组 nums 的 bitwise AND 是 nums 中所有整数的按位与结果。

* 例如，对于 nums = [1, 5, 3]，按位与等于 1 & 5 & 3 = 1。
* 此外，对于 nums = [7]，按位与为 7。

给定一个正整数候选数组 candidates 。 可以从中找若干数字形成一个组合，candidates 中的每个数字在每个组合中只能使用一次。返回按位与操作且大于 0 的组合的最大数量。

其实这道题就是考察位相关操作，我们知道在二进制按位与操作中只有都为 1 结果才为 1 ，如果出现至少一个 0 结果就为 0 ，而题目要求我们找大于 0 的组合，所以我们可以将 candidates 中所有的元素都转换成二进制，然后比较他们每个相同位上面的 1 出现的数量，最后取某个位上面的 1 的最多出现数量即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答
				

	class Solution(object):
	    def largestCombination(self, candidates):
	        """
	        :type candidates: List[int]
	        :rtype: int
	        """
	        c = [bin(c)[2:]for c in candidates]
	        maxLength = max(len(s) for s in c)
	        result = 0
	        for i in range(1, maxLength+1):
	            tmp = 0
	            for s in c:
	                if i > len(s):
	                    continue
	                elif s[-i] == '1':
	                    tmp += 1
	            result = max(result, tmp)
	        return result
            	      
			
### 运行结果

	80 / 80 test cases passed.
	Status: Accepted
	Runtime: 1492 ms
	Memory Usage: 29.1 MB


### 原题链接



https://leetcode.com/contest/weekly-contest-293/problems/largest-combination-with-bitwise-and-greater-than-zero/


您的支持是我最大的动力
