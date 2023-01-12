leetcode  2399. Check Distances Between Same Letters（python）




### 描述

You are given a 0-indexed string s consisting of only lowercase English letters, where each letter in s appears exactly twice. You are also given a 0-indexed integer array distance of length 26. Each letter in the alphabet is numbered from 0 to 25 (i.e. 'a' -> 0, 'b' -> 1, 'c' -> 2, ... , 'z' -> 25). In a well-spaced string, the number of letters between the two occurrences of the i<sup>th</sup> letter is distance[i]. If the i<sup>th</sup> letter does not appear in s, then distance[i] can be ignored.

Return true if s is a well-spaced string, otherwise return false.



Example 1:

	Input: s = "abaccb", distance = [1,3,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	Output: true
	Explanation:
	- 'a' appears at indices 0 and 2 so it satisfies distance[0] = 1.
	- 'b' appears at indices 1 and 5 so it satisfies distance[1] = 3.
	- 'c' appears at indices 3 and 4 so it satisfies distance[2] = 0.
	Note that distance[3] = 5, but since 'd' does not appear in s, it can be ignored.
	Return true because s is a well-spaced string.

	
Example 2:

	Input: s = "aa", distance = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	Output: false
	Explanation:
	- 'a' appears at indices 0 and 1 so there are zero letters between them.
	Because distance[0] = 1, s is not a well-spaced string.





Note:

	2 <= s.length <= 52
	s consists only of lowercase English letters.
	Each letter appears in s exactly twice.
	distance.length == 26
	0 <= distance[i] <= 50


### 解析

根据题意，给定一个仅由小写英文字母组成的索引为 0 的字符串 s ，其中 s 中的每个字母恰好出现两次。 还给定一个长度为 26 的 0 索引整数数组 distance 。字母表中的每个字母编号从 0 到 25（即 'a' -> 0、'b' -> 1、'c' -> 2、. .. , 'z' -> 25)。在间隔良好的字符串中，第 i<sup>th</sup> 个字母的两次出现之间的字母个数是 distance[i]。 如果第 i<sup>th</sup> 字母没有出现在 s 中，那么 distance[i] 可以忽略。如果 s 是一个间隔良好的字符串，则返回 true，否则返回 false。

其实这道题就是考察一个很简单的数组遍历，因为题目中已经给出了字符肯定会有两次，所以我们直接使用一个字典 d 将出现的字符及其位置都存好，然后只要从 a 开始遍历到 z ，只要该字符出现在 d 中（不存在的就不用管了），我们判断如果该字符在 d 中存的距离不等于 distance 存的距离，那么就直接返回 False ，否则遍历结束直接返回 True 。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def checkDistances(self, s, distance):
	        """
	        :type s: str
	        :type distance: List[int]
	        :rtype: bool
	        """
	        d = collections.defaultdict(list)
	        for i,c in enumerate(s):
	            d[c].append(i)
	        for i in range(26):
	            c = chr(i+97)
	            if c in d:
	                if distance[i] != d[c][1]- d[c][0] - 1:
	                    return False
	        return True

### 运行结果

	
	335 / 335 test cases passed.
	Status: Accepted
	Runtime: 47 ms
	Memory Usage: 13.6 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-309/problems/check-distances-between-same-letters/


您的支持是我最大的动力
