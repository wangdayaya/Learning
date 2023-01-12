leetcode  1081. Smallest Subsequence of Distinct Characters（python）

### 每日经典

《春望》 ——杜甫（唐）

国破山河在，城春草木深。

感时花溅泪，恨别鸟惊心。

烽火连三月，家书抵万金。

白头搔更短，浑欲不胜簪。

### 描述

Given a string s, return the lexicographically smallest subsequence of s that contains all the distinct characters of s exactly once.





Example 1:

	Input: s = "bcabc"
	Output: "abc"

	
Example 2:

	Input: s = "cbacdcbc"
	Output: "acdb"





Note:

	1 <= s.length <= 1000
	s consists of lowercase English letters.


### 解析

根据题意，给定一个字符串 s ，返回 s 的字典序最小子序列，其中包含 s 的所有不同的字符恰好都是一次。题意很简单，要找最小子序列说明已经先定了字符之间的先后顺序，要保证字典序最小那就尽量按照字符生序排列，而且要包含所有出现的字符。

那我们我们可以从左往右按顺序遍历字符，将字符放入一个字符递增的单调字符序列 q ，这样可以尽量保证字典序最小， 如果遍历到某个字符已经使用过了，就直接进入下一个字符的遍历；如果没出现过，判断栈顶的字符是否后面还会出现且栈顶字符大于当前字符，如果是的话就将 q 的栈顶字符一直去掉，按照上面的思路处理，将最后得到的 q 转换成字符串返回即可。


### 解答
				

	class Solution(object):
	    def smallestSubsequence(self, text):
	        """
	        :type s: str
	        :rtype: str
	        """
	        d = collections.defaultdict(int)
	        for c in text:
	            d[c] += 1
	        s = set()
	        q = []
	        for c in text:
	            d[c] -= 1
	            if c not in s:
	                while q and d[q[-1]]>0 and q[-1]>c:
	                    s.remove(q.pop())
	                q.append(c)
	                s.add(c)
	        return ''.join(q)
	                    
	        
            	      
			
### 运行结果

	Runtime: 24 ms, faster than 51.85% of Python online submissions for Smallest Subsequence of Distinct Characters.
	Memory Usage: 13.6 MB, less than 44.44% of Python online submissions for Smallest Subsequence of Distinct Characters.



原题链接：https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/



您的支持是我最大的动力
