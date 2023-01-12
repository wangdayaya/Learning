leetcode 2182. Construct String With Repeat Limit （python）


### 前言

这是 Weekly Contest 281 的第三题，难度 Medium ，考查的就是对堆数据结构的理解，难度也不是很难，但是比赛的时候没有做出来，烦。

### 描述

You are given a string s and an integer repeatLimit. Construct a new string repeatLimitedString using the characters of s such that no letter appears more than repeatLimit times in a row. You do not have to use all characters from s.

Return the lexicographically largest repeatLimitedString possible.

A string a is lexicographically larger than a string b if in the first position where a and b differ, string a has a letter that appears later in the alphabet than the corresponding letter in b. If the first min(a.length, b.length) characters do not differ, then the longer string is the lexicographically larger one.



Example 1:


	Input: s = "cczazcc", repeatLimit = 3
	Output: "zzcccac"
	Explanation: We use all of the characters from s to construct the repeatLimitedString "zzcccac".
	The letter 'a' appears at most 1 time in a row.
	The letter 'c' appears at most 3 times in a row.
	The letter 'z' appears at most 2 times in a row.
	Hence, no letter appears more than repeatLimit times in a row and the string is a valid repeatLimitedString.
	The string is the lexicographically largest repeatLimitedString possible so we return "zzcccac".
	Note that the string "zzcccca" is lexicographically larger but the letter 'c' appears more than 3 times in a row, so it is not a valid repeatLimitedString.
	



Note:


	1 <= repeatLimit <= s.length <= 10^5
	s consists of lowercase English letters.

### 解析


根据题意，给定一个字符串 s 和一个整数 repeatLimit。 使用 s 的字符构造一个新的字符串 repeatLimitedString，使得没有字母连续出现超过 repeatLimit 次。 返回可能的字典序最大的 repeatLimitedString。另外还给出了字典序的定义，说人话就是有 abcdefg...xyz 这 26 个字母，尽量将后面的字母往字符串的开头放就行。

比赛的时候我的思路一直都在尝试使用单调栈来解题，但是绕来绕去都写不出代码，最后只能放弃去做第四题去了。看了论坛的大佬解题思路，豁然开朗，其中一种解法就是使用了堆数据结构。我们尽量在满足 repeatLimit 的条件下，将字典序较大的字母放入 result 中，如果超过 repeatLimit 个，则之后放一个字典序稍微小的字母，遍历堆结束，就可以将 result 转换成字符串返回。

时间复杂度是 O(N) ，因为用到了堆排序，空间复杂度为 O(1) ，因为最多只有 26 个字母。


### 解答
				
	class Solution(object):
	    def repeatLimitedString(self, s, repeatLimit):
	        """
	        :type s: str
	        :type repeatLimit: int
	        :rtype: str
	        """
	        heap = [(-ord(k), k, v) for k, v in collections.Counter(s).items()]
	        heapq.heapify(heap)
	        result = []
	        count = 0
	        while heap:
	            order, c, remaining = heapq.heappop(heap)
	            if result and result[-1] == c and (count + 1) > repeatLimit:
	                if not heap: break
	                order2, c2, remaining2 = heapq.heappop(heap)
	                heapq.heappush(heap, (order, c, remaining))
	                order, c, remaining = order2, c2, remaining2
	            if result and result[-1] != c:
	                count = 0
	            result.append(c)
	            remaining -= 1
	            if remaining:
	                heapq.heappush(heap, (order, c, remaining))
	            count += 1
	        return ''.join(result)

            	      
			
### 运行结果


	150 / 150 test cases passed.
	Status: Accepted
	Runtime: 2520 ms
	Memory Usage: 17.1 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-281/problems/construct-string-with-repeat-limit/



您的支持是我最大的动力
