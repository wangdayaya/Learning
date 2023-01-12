「这是我参与2022首次更文挑战的第N天，活动详情查看：[2022首次更文挑战](https://juejin.cn/post/7052884569032392740 "https://juejin.cn/post/7052884569032392740")」。

### 前言
这是 leetcode 中 Biweekly Contest 72 的第三题，难度为 Medium ，考查的就是贪心思想，也不难。

### 描述


You are given an integer finalSum. Split it into a sum of a maximum number of unique positive even integers.

* For example, given finalSum = 12, the following splits are valid (unique positive even integers summing up to finalSum): (2 + 10), (2 + 4 + 6), and (4 + 8). Among them, (2 + 4 + 6) contains the maximum number of integers. Note that finalSum cannot be split into (2 + 2 + 4 + 4) as all the numbers should be unique.

Return a list of integers that represent a valid split containing a maximum number of integers. If no valid split exists for finalSum, return an empty list. You may return the integers in any order.


Example 1:


	Input: finalSum = 12
	Output: [2,4,6]
	Explanation: The following are some valid splits: (2 + 10), (2 + 4 + 6), and (4 + 8).
	(2 + 4 + 6) has the maximum number of integers, which is 3. Thus, we return [2,4,6].
	Note that [2,6,4], [6,2,4], etc. are also accepted.
	
Example 2:

	Input: finalSum = 7
	Output: []
	Explanation: There are no valid splits for the given finalSum.
	Thus, we return an empty array.



Note:


* 1 <= finalSum <= 10^10

### 解析


根据题意，给定一个整数 finalSum。 将其拆分为数量最多的唯一正偶数之和。

例如，给定 finalSum = 12，以下拆分是有效的（唯一的正偶数总和为 finalSum）：(2 + 10)、(2 + 4 + 6) 和 (4 + 8)。 其中，(2 + 4 + 6) 包含最大整数个数。 请注意，finalSum 不能拆分为 (2 + 2 + 4 + 4)，因为所有数字都应该是唯一的。

返回一个整数列表，该列表表示包含最大整数数量的有效拆分。 如果 finalSum 不存在有效拆分，则返回一个空列表。 结果可以按任何顺序返回整数。

其实想要拆分的越多越好，那就尽量使用越小的偶数进行组合，首先我们需要考虑的是什么情况下的 finalSum 没有结果？那就是当 finalSum 为奇数的时候，因为奇数是无法通过都为偶数的数字进行求和得到的，所以当 finalSum 为奇数的时候我们直接返回空列表即可。

对于 finalSum 为偶数的情况下，我们可以使用贪心的思想，尽量每次取最小的偶数去操作，换句话说就是我们使用 while 循环，用从 2 开始的偶数对 finalSum 进行减法操作，将每个偶数加入 result 中，当 finalSum <= 0 的时候跳出循环，当 finalSum 为 0 的时候说明我们是尽可能取到了尽可能小的且最多数量的偶数，直接返回 result 即可。当 finalSum<0 的时候，说明我们需要从 result 中移除 -finalSum 这个数字才能满足 result 中元素和为 finalSum 的条件，所以我们将  -finalSum 从 result 中移除，然后返回 result 即可。

### 解答
				

	class Solution(object):
	    def maximumEvenSplit(self, finalSum):
	        """
	        :type finalSum: int
	        :rtype: List[int]
	        """
	        if finalSum%2!=0: return []
	        result = []
	        count = 2
	        while finalSum>0:
	            result.append(count)
	            finalSum -= count
	            count += 2
	        if finalSum == 0: return result
	        if finalSum < 0:
	            result.remove(-finalSum)
	            return result
	            
	            
	        
            	      
			
### 运行结果

	56 / 56 test cases passed.
	Status: Accepted
	Runtime: 532 ms
	Memory Usage: 26.3 MB


### 原题链接


https://leetcode.com/contest/biweekly-contest-72/problems/maximum-split-of-positive-even-integers/


您的支持是我最大的动力
