leetcode  2030. Smallest K-Length Subsequence With Occurrences of a Letter（python）

### 每日经典

《狱中题壁》 ——谭嗣同（清）

望门投止思张俭，忍死须臾待杜根。

我自横刀向天笑，去留肝胆两昆仑。

### 描述

You are given a string s, an integer k, a letter letter, and an integer repetition.

Return the lexicographically smallest subsequence of s of length k that has the letter letter appear at least repetition times. The test cases are generated so that the letter appears in s at least repetition times.

A subsequence is a string that can be derived from another string by deleting some or no characters without changing the order of the remaining characters.

A string a is lexicographically smaller than a string b if in the first position where a and b differ, string a has a letter that appears earlier in the alphabet than the corresponding letter in b.



Example 1:

	Input: s = "leet", k = 3, letter = "e", repetition = 1
	Output: "eet"
	Explanation: There are four subsequences of length 3 that have the letter 'e' appear at least 1 time:
	- "lee" (from "leet")
	- "let" (from "leet")
	- "let" (from "leet")
	- "eet" (from "leet")
	The lexicographically smallest subsequence among them is "eet".

	
Example 2:

![](https://assets.leetcode.com/uploads/2021/09/13/smallest-k-length-subsequence.png)

	Input: s = "leetcode", k = 4, letter = "e", repetition = 2
	Output: "ecde"
	Explanation: "ecde" is the lexicographically smallest subsequence of length 4 that has the letter "e" appear at least 2 times.


Example 3:

	
	Input: s = "bb", k = 2, letter = "b", repetition = 2
	Output: "bb"
	Explanation: "bb" is the only subsequence of length 2 that has the letter "b" appear at least 2 times.
	



Note:

	1 <= repetition <= k <= s.length <= 5 * 10^4
	s consists of lowercase English letters.
	letter is a lowercase English letter, and appears in s at least repetition times.



### 解析

根据题意，给定一个字符串 s 、一个整数 k 、一个字母 letter 和一个整数 repetition 。返回 letter 出现至少 repetition 次数的长度为 k 的 s 的字典序最小子序列。 这道题的关键点有三个，第一个是返回的长度为 k 的子序列，第二个是必须包含 letter 字母，第三个是结果字符串必须包含 repetition 个 letter ，第四要保证得到的结果是字典序最小的。其实看到这类题我们基本就能判断出来使用单调栈的思路来解题，维持一个字典序递增的单调栈 stack ，只不过是限制条件比较多，要保证前面提到的三个要求即可。
        


### 解答
				
	class Solution(object):
	    def smallestSubsequence(self, s, k, letter, repetition):
	        c0 = len(s) - k   # 最多删除字符个数
	        c1 = s.count(letter) - repetition # 最多删除的 letter 个数
	        stack = []
	        c2 = 0 # 统计删除了几个字符
	        c3 = 0 # 统计删除了几个 letter
	        for c in s:
	            while stack and stack[-1]>c and c2<c0 and (stack[-1]!=letter or (stack[-1]==letter and c3<c1)):
	                if stack[-1]==letter:
	                    c3 += 1
	                c2 += 1
	                stack.pop()
	            stack.append(c)
	        result = ''
	        for i in range(len(stack)-1, -1, -1):
	            if c2==c0 or (stack[i]==letter and c1==c3):
	                result += stack[i]
	            else:
	                c2 += 1
	                if stack[i] == letter:
	                    c3 += 1
	        return result[::-1]
	        
	            
			
### 运行结果


	Runtime: 2736 ms, faster than 6.67% of Python online submissions for Smallest K-Length Subsequence With Occurrences of a Letter.
	Memory Usage: 16.6 MB, less than 6.67% of Python online submissions for Smallest K-Length Subsequence With Occurrences of a Letter.

原题链接：https://leetcode.com/problems/smallest-k-length-subsequence-with-occurrences-of-a-letter/



您的支持是我最大的动力
