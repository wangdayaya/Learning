leetcode 2310. Sum of Numbers With Units Digit K （python）




### 描述

Given two integers num and k, consider a set of positive integers with the following properties:

The units digit of each integer is k.
The sum of the integers is num.
Return the minimum possible size of such a set, or -1 if no such set exists.

Note:

The set can contain multiple instances of the same integer, and the sum of an empty set is considered 0.
The units digit of a number is the rightmost digit of the number.



Example 1:

	Input: num = 58, k = 9
	Output: 2
	Explanation:
	One valid set is [9,49], as the sum is 58 and each integer has a units digit of 9.
	Another valid set is [19,39].
	It can be shown that 2 is the minimum possible size of a valid set.

	
Example 2:

	Input: num = 37, k = 2
	Output: -1
	Explanation: It is not possible to obtain a sum of 37 using only integers that have a units digit of 2.


Example 3:

	Input: num = 0, k = 7
	Output: 0
	Explanation: The sum of an empty set is considered 0.

	


Note:

	0 <= num <= 3000
	0 <= k <= 9



### 解析


根据题意，给定两个整数 num 和 k ，考虑一个包含以下性质的正整数集合：

* 每个整数的个位为 k
* 整数之和为 num

返回此类集合的最小可能的大小，如果不存在此类集合，则返回 -1 。需要注意的是，集合可以包含同一个整数的多个实例，空集合的和被认为是 0 。

这道题可以算作一个数学题，我们假设现在集合中有个 n 个整数，那么根据上面的两个条件，我们可以知道如下公式：

* 	n<sub>1</sub> + n<sub>2</sub> + ... + n<sub>n</sub> = num

因为每个正数的个位都是 k ，那么可以变为：

* 	(k+10\*a<sub>1</sub>) + (k+10\*a<sub>2</sub>) + ... + (k+10\*a<sub>n</sub>) = n\*k + 10\*(a<sub>1</sub>+a<sub>2</sub>+...+a<sub>n</sub>) = num

得到上面的公式，就可以得到这个 num-n\*k 肯定是 10 的倍数，所以我们遍历从 1 到 10 ，如果存在答案肯定会找到，否则直接返回 -1 。

为什么知道如果有结果会在 1 到 10 之间，因为我们是对 10 做取模操作的呀。

时间复杂度为 O(1) ，空间复杂度为 O(1) 。

### 解答
				
	class Solution(object):
	    def minimumNumbers(self, num, k):
	        """
	        :type num: int
	        :type k: int
	        :rtype: int
	        """
	        if num == 0 :return 0
	        for i in range(1, 11):
	            if (num-i*k) % 10 == 0 and i*k <= num:
	                return i
	        return -1

            	      
			
### 运行结果


	298 / 298 test cases passed.
	Status: Accepted
	Runtime: 43 ms
	Memory Usage: 13.3 MB


### 原题链接


https://leetcode.com/contest/weekly-contest-298/problems/sum-of-numbers-with-units-digit-k/


您的支持是我最大的动力
