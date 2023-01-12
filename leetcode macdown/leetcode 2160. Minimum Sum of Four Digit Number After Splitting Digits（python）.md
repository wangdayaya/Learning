2160. Minimum Sum of Four Digit Number After Splitting Digits

### 前言

这是 Biweekly Contest 71 比赛的第一题，难度 Easy ，考察的是对题目的数组的操作，很简单。


### 描述


You are given a positive integer num consisting of exactly four digits. Split num into two new integers new1 and new2 by using the digits found in num. Leading zeros are allowed in new1 and new2, and all the digits found in num must be used.

* For example, given num = 2932, you have the following digits: two 2's, one 9 and one 3. Some of the possible pairs [new1, new2] are [22, 93], [23, 92], [223, 9] and [2, 329].

Return the minimum possible sum of new1 and new2.


Example 1:

	Input: num = 2932
	Output: 52
	Explanation: Some possible pairs [new1, new2] are [29, 23], [223, 9], etc.
	The minimum sum can be obtained by the pair [29, 23]: 29 + 23 = 52.



Note:

	1000 <= num <= 9999


### 解析


根据题意，给你一个正整数 num 正好由四位数字组成。 使用 num 中的数字将 num 拆分为两个新整数 new1 和 new2 。 new1 和 new2 中允许使用前导零，并且必须使用在 num 中找到的所有数字。例如，给定 num = 2932，会有以下数字：两个 2，一个 9 和一个 3 。一些可能的对 [new1, new2] 是 [22, 93], [23, 92], [223, 9 ] 和 [2, 329]。返回 new1 和 new2 的最小可能总和。

这种题其实可以用最朴素的算法来求解，因为题目中的限制条件 num 只是个四位数，所以我们先把各个位置上面的数字都提取出来放入一个列表 digits 中，然后进行升序排序，因为要想让 digits 中的所有数字组成的两个数字的和最小，那就是要尽量让最前面的数字尽量和最后面的数字搭配组成新的数字。如果这个四位数的四个数字都不为 0 ，可以这样操作，但是如果有 0 ，就不一样了，我们要将 digits 中的 0 都弹 pop 出去，这样就不会影响我们的计算，此时有四种情况：

* 当 digits 的长度还是 4 ，那么直接按照上面的算法返回 
	
		digits[0]*10+digits[-1] +digits[1]*10+digits[2]

* 当 digits 的长度还是 3 ，那么让前两个数字组成一个数，再加上第三个数字和最小，返回

		digits[0] * 10 + digits[1] + digits[2]

* 当 digits 的长度还是 2 ，那么每个数字自成一个数的和最小，那么返回

		digits[0] + digits[1]
* 当 digits 的长度还是 1 ，那么直接返回
	
		digits[0]

因为 num 只是个四位数，所以时间复杂度和空间复杂度可以都看作 O(1) 。


### 解答
				

	class Solution(object):
	    def minimumSum(self, num):
	        """
	        :type num: int
	        :rtype: int
	        """
	        digits = []
	        for i in str(num):
	            digits.append(int(i))
	        digits.sort()
	        while digits:
	            if digits[0] == 0:
	                digits.pop(0)
	            else:
	                break
	        N = len(digits)
	        if N == 4:
	            return digits[0]*10+digits[-1] +digits[1]*10+digits[2]
	        elif N == 3:
	            return digits[0] * 10 + digits[1] + digits[2]
	        elif N == 2:
	            return digits[0] + digits[1]
	        elif N == 1:
	            return digits[0]
	                
	        
	        
	        
            	      
			
### 运行结果

	99 / 99 test cases passed.
	Status: Accepted
	Runtime: 20 ms
	Memory Usage: 13.4 MB


### 原题链接

https://leetcode.com/contest/biweekly-contest-71/problems/minimum-sum-of-four-digit-number-after-splitting-digits/

您的支持是我最大的动力
