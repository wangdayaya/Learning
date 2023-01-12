leetcode 2217. Find Palindrome With Fixed Length （python）




### 描述


Given an integer array queries and a positive integer intLength, return an array answer where answer[i] is either the queries[i]th smallest positive palindrome of length intLength or -1 if no such palindrome exists.

A palindrome is a number that reads the same backwards and forwards. Palindromes cannot have leading zeros.


Example 1:


	Input: queries = [1,2,3,4,5,90], intLength = 3
	Output: [101,111,121,131,141,999]
	Explanation:
	The first few palindromes of length 3 are:
	101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, ...
	The 90th palindrome of length 3 is 999.
	
Example 2:

	Input: queries = [2,4,6], intLength = 4
	Output: [1111,1331,1551]
	Explanation:
	The first six palindromes of length 4 are:
	1001, 1111, 1221, 1331, 1441, and 1551.



Note:

	1 <= queries.length <= 5 * 10^4
	1 <= queries[i] <= 10^9
	1 <= intLength <= 15


### 解析

根据题意，给定一个整数数组 queries 和一个正整数 intLength，返回一个数组 answer ，其中 answer[i] 是第 queries[i] 个长度为 intLength 的最小正回文数，如果不存在这样的回文数，则返回 -1 。题目中还给出了回文数的定义，回文数是一个前后读相同的数字。 回文不能有前导零。


限制条件中 intLength 的长度最大为 15 ，说明我们不可能通过暴力求解得到所有的回文数，然后再通过 queries 找出所有的结果。其实如果知道回文数的规律，这道题可以直接通过规律来找出所有的答案。

当 intLength 为偶数的时候，假如为 4 ，我们只要找到前 2 个数字的最小数 10  ，最大为 99 ，然后把找到的数字进行逆序反转拼接到前两个数字后面就是最后的结果，而且如果前 2 个数字按照升序排序，那么最后拼接而成的回文数也是升序的，所以我们可以直接将 10  加 queries[i] 获得指定大小的回文数。

当 intLength 为奇数的时候，假如为 3 ，我们就暂时将其当作长度为 4 的大小，同理前两个数字去找 10 到 99 范围内的数字 n ，但是此时的后半部分需要拼接的却是 n[::-1][1:] ，因为 intLength 是奇数，所以我们可以直接将 10 加 queries[i] 获得指定大小的回文数。

时间复杂度为 O(N)，空间复杂度为 O(1) 。


### 解答
				

	class Solution(object):
	    def kthPalindrome(self, queries, intLength):
	        """
	        :type queries: List[int]
	        :type intLength: int
	        :rtype: List[int]
	        """
	        origin = intLength
	        if intLength % 2:
	            intLength += 1
	        k = pow(10, intLength // 2 - 1)
	        result = []
	        for q in queries:
	            tmp = str(k + q - 1)
	            if origin % 2:
	                tmp += tmp[::-1][1:]
	            else:
	                tmp += tmp[::-1]
	            if len(tmp) == origin:
	                result.append(int(tmp))
	            else:
	                result.append(-1)
	        return result
            	      
			
### 运行结果

	
	162 / 162 test cases passed.
	Status: Accepted
	Runtime: 1042 ms
	Memory Usage: 23.3 MB


### 原题链接



https://leetcode.com/contest/weekly-contest-286/problems/find-palindrome-with-fixed-length/


您的支持是我最大的动力
