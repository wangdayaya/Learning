leetcode 567. Permutation in String （python）




### 描述


Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise.

In other words, return true if one of s1's permutations is the substring of s2.




Example 1:

	Input: s1 = "ab", s2 = "eidbaooo"
	Output: true
	Explanation: s2 contains one permutation of s1 ("ba").

	
Example 2:


	Input: s1 = "ab", s2 = "eidboaoo"
	Output: false






Note:

	1 <= s1.length, s2.length <= 10^4
	s1 and s2 consist of lowercase English letters.


### 解析
根据题意， 给定两个字符串 s1 和 s2，如果 s2 包含 s1 的排列，则返回 true，否则返回 false。换句话说，如果 s1 的排列之一是 s2 的子字符串，则返回 true。

这道题的题意简单明了，是个典型的滑动窗口的问题，只不过我们不能直接滑动窗口进行判断，这样会超时，我们这里借用了字典数据结构来进行判断。因为如果 s1 的排列之一是 s2 的子字符串，那么 s1 中每个字符及其对应的出现次数，肯定和 s2 中的某个子字符串中每个字符及其对应的出现次数相同，所以先对 s1 中的每个字符进行计数，存储到字典 c 中，然后我们从左往右依次遍历窗口长度和 s1 相等的 s2 的子字符串，然后判断该子字符串的字母出现次数和 c 是否相同，如果相等则直接返回 True ，如果遍历结束直接返回 False 。

其实这种解法刚好能通过，但是耗时也很高，因为需要不断的去对子字符串计数时间复杂度是 O(l1+l1\*(l2-l1+1)) ，几乎就是 O(N^2)，这里是因为限制条件 s1 和 s2 的最长长度为 10^4 ，险过测试用例罢了，如果是 10^5 肯定超时了，空间复杂度为 O(26*2) ，也就是 O(1) 。
### 解答
				

	class Solution(object):
	    def checkInclusion(self, s1, s2):
	        """
	        :type s1: str
	        :type s2: str
	        :rtype: bool
	        """
	        c = collections.Counter(s1)
	        N = len(s1)
	        for i in range(len(s2)-N+1):
	            if self.isPermutation(c, s2[i:i+N]):
	                return True
	        return False
	    
	    def isPermutation(self, c, s):
	        t = collections.Counter(s)
	        if t == c:
	            return True
	        return False
            

			
### 运行结果


	Runtime: 8703 ms, faster than 6.87% of Python online submissions for Permutation in String.
	Memory Usage: 13.4 MB, less than 99.34% of Python online submissions for Permutation in String.

### 解析

我们可以对上面的滑动窗口进行优化，因为我们上面主要的耗时就是不断对 s2 的子字符串进行字典计数，我们这次将这一步进行优化，先初始化两个字典 c 和 t ，每个字典都保存了 26 个字母及其出现的个数，然后每次滑动窗口只是将当前窗口滑过去消失的字母出现次数减一，对新进来的字母次数加一即可。然后比较 c 和 t 是否相等，如果相等直接返回 True ，如果遍历结束直接返回 False ，这样我们的时间复杂度会降到 O(N) ，空间复杂度还是 O(1) 。看最后的耗时，从 8703ms 骤降到 101ms ，是不是算法很神奇！

### 解答
	class Solution(object):
	    def checkInclusion(self, s1, s2):
	        """
	        :type s1: str
	        :type s2: str
	        :rtype: bool
	        """
	        c = collections.Counter(s1)
	        N = len(s1)
	        t = collections.Counter(s2[:N])
	        if c==t: return True
	        for i in range(26):
	            letter = chr(ord('a')+i)
	            if letter not in c:
	                c[letter] = 0
	            if letter not in t:
	                t[letter] = 0
	        for i in range(1, len(s2)-N+1):
	            t[s2[i-1]] -= 1
	            t[s2[i+N-1]] += 1
	            if c == t:
	                return True
	        return False
### 运行结果

	Runtime: 101 ms, faster than 53.87% of Python online submissions for Permutation in String.
	Memory Usage: 14.1 MB, less than 18.35% of Python online submissions for Permutation in String.

### 原题链接


https://leetcode.com/problems/permutation-in-string/


您的支持是我最大的动力
