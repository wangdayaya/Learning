leetcode  91. Decode Ways（python）




### 描述


A message containing letters from A-Z can be encoded into numbers using the following mapping:
	
	'A' -> "1"
	'B' -> "2"
	...
	'Z' -> "26"
	
To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, "11106" can be mapped into:

* "AAJF" with the grouping (1 1 10 6)
* "KJF" with the grouping (11 10 6)

Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06". Given a string s containing only digits, return the number of ways to decode it. The test cases are generated so that the answer fits in a 32-bit integer.


Example 1:

	Input: s = "12"
	Output: 2
	Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).


	
Example 2:

	Input: s = "226"
	Output: 3
	Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).


Example 3:

	Input: s = "06"
	Output: 0
	Explanation: "06" cannot be mapped to "F" because of the leading zero ("6" is different from "06").



Note:

	1 <= s.length <= 100
	s contains only digits and may contain leading zero(s).


### 解析

根据题意，一条包含字母 A-Z 的消息通过以下映射方式进行了编码 ：

	'A' -> "1"
	'B' -> "2"
	...
	'Z' -> "26"
	
要解码已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，"11106" 可以映射为：

	"AAJF" ，将消息映射回 (1 1 10 6)
	"KJF" ，将消息映射回 (11 10 6)

注意，消息不能映射为  (1 11 06) ，因为 "06" 不能映射为 "F" ，这是由于 "6" 和 "06" 在映射中并不等价。给定一个只含数字的非空字符串 s ，请计算并返回解码方法的总数。题目保证答案肯定是一个 32 位的整数。


首先这道题使用暴力方法进行递归是会超时的，因为每个题目中规定 s 长度最大为 100 ，而对于每一个字符 s[i] ，不仅自己可能是合法的存在，其和 s[i-1] 也可能是一种合法的存在，另外在递归过程中会有很多重复的计算，所以这种暴力的解法时间复杂度是 O(2^N) ，N 是 s 的长度，我们可以考虑使用动态规划来进行解题。
 
 我们定义动态规划数组 dp ，dp[i] 表示前 i 个字符有多少种解码方案。其实对于任意一个字符 s[i] 最多和前面一个字符 s[i-1] 有关系，所以对于两个字符而言，可能有 0 种、1 种、2 种解码方案。我们首先定义了正确的数字字符串集合 nums ，这可以帮助我们检查数字是否合法，然后具体分析各种情况：
 
（1） s[i] 不在 nums 中，相当于 s[i] 为 0 的情况：
 
* s[i-1] + s[i] 不在 nums 中，表示无法解码，直接返回 0 即可
* s[i-1] + s[i] 在 nums 中，这时只有一种解码可能，所以 dp[i] 等于 dp[i-2] 
 
（2）s[i] 在 nums 中：
 
*  s[i-1]+s[i] 不在 nums 中，表示无法解码，此时 s[i] 只能当作单个字符进行解码，所以 dp[i] 等于 dp[i-1] 
*   s[i-1]+s[i] 在 nums 中，此时两个字符可以分别进行解码，也可以合起来进行解码，所以 dp[i] 等于 dp[i-2] + dp[i-1] 

在初始化 dp 的时候，因为我们需要 dp[i-2] ，所以要先对 dp[0] 、dp[1] 进行初始化，并且对 dp[1] 的各种可能性进行分析。

时间复杂度为 O(N) ，空间复杂度为 O(N) ，N 为字符串的长度。


### 解答

	class Solution:
	    def numDecodings(self, s: str) -> int:
	        if s[0] == '0': return 0
	        if len(s) == 1: return 1
	        nums = set(str(i) for i in range(1, 27))
	        dp = [0] * (len(s))
	        dp[0] = 1
	        if s[1] not in nums:
	            dp[1] = 1 if s[: 2] in nums else 0
	        else:
	            dp[1] = 2 if s[: 2] in nums else 1
	        for i in range(2, len(s)):
	            if s[i] not in nums:
	                if s[i - 1: i + 1] not in nums:
	                    return 0
	                else:
	                    dp[i] = dp[i - 2]
	            else:
	                if s[i - 1: i + 1] in nums:
	                    dp[i] = dp[i - 1] + dp[i - 2]
	                else:
	                    dp[i] = dp[i - 1]
	        return dp[-1]

### 运行结果

	Runtime: 64 ms, faster than 29.63% of Python3 online submissions for Decode Ways.
	Memory Usage: 13.9 MB, less than 80.35% of Python3 online submissions for Decode Ways.

### 原题链接

https://leetcode.com/problems/decode-ways/


您的支持是我最大的动力
