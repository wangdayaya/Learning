leetcode 3. Longest Substring Without Repeating Characters （python）



### 描述

Given a string s, find the length of the longest substring without repeating characters.





Example 1:

	Input: s = "abcabcbb"
	Output: 3
	Explanation: The answer is "abc", with the length of 3.

	
Example 2:

	Input: s = "bbbbb"
	Output: 1
	Explanation: The answer is "b", with the length of 1.


Example 3:


	Input: s = "pwwkew"
	Output: 3
	Explanation: The answer is "wke", with the length of 3.
	Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
	



Note:

	0 <= s.length <= 5 * 10^4
	s consists of English letters, digits, symbols and spaces.


### 解析

根据题意， 给定一个字符串 s ，找出含有不重复字符的最长子字符串的长度。

这道题的题意是简洁明了的，而且我们只要有经验，碰到这种题为了保证时间复杂度符合 AC 要求，肯定要使用滑动窗口的解法，这样才能在 O(N) 时间复杂度下完成解题。方向我们已经有了，具体思路如下：

* 定义 result 为最终结果， L 为滑动窗口的最左端索引，d 为保存每个遍历过的的字符的索引位置
* 遍历 s 中所有的字符 c 其索引为 i ，如果这个字符没有出现在 d 中，说明当前从 L 到 i 的子字符串都没有重复的，我们直接使用此子字符串的长度更新 result ；如果 c 已经出现在了 d 中，并且 d[c] >= L ，那么说明这个 c 字符出现在了从 L 到 i 的子字符串中，此时我们将 L 更新为 d[c] + 1 即可，继续后面的更新操作。每次遍历更新每个字符最新的索引 d[c] = i 。
* 遍历结束直接返回 result 即可

时间复杂度为 O(N) ，空间复杂度为 O(字符集合大小) 。

### 解答
				
	class Solution(object):
	    def lengthOfLongestSubstring(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        result = 0
	        L = 0
	        d = {}
	        for i, c in enumerate(s):
	            if c in d and d[c] >= L:
	                L = d[c] + 1
	            else:
	                result = max(result, i - L + 1)
	            d[c] = i
	        return result

            	      
			
### 运行结果

	Runtime: 38 ms, faster than 94.63% of Python online submissions for Longest Substring Without Repeating Characters.
	Memory Usage: 13.8 MB, less than 62.00% of Python online submissions for Longest Substring Without Repeating Characters.


### 原题链接

https://leetcode.com/problems/longest-substring-without-repeating-characters/

您的支持是我最大的动力
