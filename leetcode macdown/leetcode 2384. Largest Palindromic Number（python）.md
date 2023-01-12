leetcode  2384. Largest Palindromic Number（python）




### 描述

You are given a string num consisting of digits only. Return the largest palindromic integer (in the form of a string) that can be formed using digits taken from num. It should not contain leading zeroes.

Notes:

* You do not need to use all the digits of num, but you must use at least one digit.
* The digits can be reordered.



Example 1:

	Input: num = "444947137"
	Output: "7449447"
	Explanation: 
	Use the digits "4449477" from "444947137" to form the palindromic integer "7449447".
	It can be shown that "7449447" is the largest palindromic integer that can be formed.

	
Example 2:

	Input: num = "00009"
	Output: "9"
	Explanation: 
	It can be shown that "9" is the largest palindromic integer that can be formed.
	Note that the integer returned should not contain leading zeroes.



Note:

	1 <= num.length <= 10^5
	num consists of digits.


### 解析

根据题意，给定一个字符串 num 仅由数字组成。 返回可以使用从 num 中获取的数字形成的最大回文整数（以字符串的形式）。 它不应包含前导零。注意：

* 不需要使用 num 的所有数字，但必须至少使用一位。
* 数字可以重新排序。

其实这道题的关键就在几个关键字上“最大”、“回文”、“不能有前导零”，明显是个考察贪心的题目，要想使回文数最大，我们先考虑左半部分 left ，因为右半部分是镜像的，从左到右每个位置上放置的肯定是当前可用的并且其出现的次数大于 1 的最大数字，所以我们只需要把这些数字按照其出现的次数除 2 的个数挨个加入到 left 中即可，然后当 left 不为空的时候，此时满足不能有前导零的要求，所以如果有可用的 0 我们可以将其按照出现的次数除 2 的个数加入到 left 后面，如果此时剩下的还有出现次数为 1 的数字可用，那么我们将最大的数字拼接到 left 后面即可，最后我们按照镜像拼接左右字符串即可得到最大的可能结果。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def largestPalindromic(self, num):
	        """
	        :type num: str
	        :rtype: str
	        """
	        cnt = collections.Counter(num)
	        if cnt['0'] == len(num):
	            return "0"
	        left = ""
	        for c in digits[:0:-1]:
	            left += c * (cnt[c] // 2)
	        if left:
	            left += '0' * (cnt['0'] // 2)
	
	        right = left[::-1]
	        for c in digits[::-1]:
	            if cnt[c] % 2:
	                left += c
	                break
	        return left + right

### 运行结果

	67 / 67 test cases passed.
	Status: Accepted
	Runtime: 138 ms
	Memory Usage: 15.3 MB
	Submitted: 0 minutes ago


### 原题链接

	https://leetcode.com/contest/weekly-contest-307/problems/largest-palindromic-number/


您的支持是我最大的动力
