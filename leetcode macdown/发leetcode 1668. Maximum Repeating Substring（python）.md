leetcode 1668. Maximum Repeating Substring （python）

### 描述

For a string sequence, a string word is k-repeating if word concatenated k times is a substring of sequence. The word's maximum k-repeating value is the highest value k where word is k-repeating in sequence. If word is not a substring of sequence, word's maximum k-repeating value is 0.

Given strings sequence and word, return the maximum k-repeating value of word in sequence.



Example 1:

	Input: sequence = "ababc", word = "ab"
	Output: 2
	Explanation: "abab" is a substring in "ababc".

	
Example 2:


	Input: sequence = "ababc", word = "ba"
	Output: 1
	Explanation: "ba" is a substring in "ababc". "baba" is not a substring in "ababc".

Example 3:

	Input: sequence = "ababc", word = "ac"
	Output: 0
	Explanation: "ac" is not a substring in "ababc". 
	



Note:

	1 <= sequence.length <= 100
	1 <= word.length <= 100
	sequence and word contains only lowercase English letters.


### 解析

根据题意，就是找出字符串 sequence 中最大个数的连续 word 。思路比较简单，就是先计算 word 可以连续拼接的最大次数 n ，然后将 word 拼接 n 次之后判断是否存在于 sequence 中，如果不存在那就判断 n-1 是否存在于 sequence 中，以此类推，如果都没有，那就直接返 0 即可。


### 解答
				
	class Solution(object):
	    def maxRepeating(self, sequence, word):
	        """
	        :type sequence: str
	        :type word: str
	        :rtype: int
	        """
	        L = len(sequence)
	        l = len(word)
	        n = L // l
	        for i in range(n, 0, -1):
	            if sequence.count(word * i):
	                return i
	        return 0
	        
            	      
			
### 运行结果

	Runtime: 16 ms, faster than 76.23% of Python online submissions for Maximum Repeating Substring.
	Memory Usage: 13.4 MB, less than 59.84% of Python online submissions for Maximum Repeating Substring.

### 解析

另外，可以用 python 的内置函数 find 来解决该题。基本的思想和上面的类似，就是计算出 word 可能存在于 sequence 的最大的次数，然后遍历 [1,count] 中的每个元素 n ，使用 find 函数查找 word*n 是否存在于 sequence ，如果存在则结果加一，否则直接返回 result 。

### 解答

	class Solution(object):
	    def maxRepeating(self, sequence, word):
	        """
	        :type sequence: str
	        :type word: str
	        :rtype: int
	        """
	        result = 0
	        count = len(sequence) // len(word)
	        for n in range(1,  count + 1):
	            if sequence.find(word * n) != -1:
	                result += 1
	            else:
	                break
	        return result

### 运行结果


	Runtime: 8 ms, faster than 100.00% of Python online submissions for Maximum Repeating Substring.
	Memory Usage: 13.6 MB, less than 34.78% of Python online submissions for Maximum Repeating Substring.
	
原题链接：https://leetcode.com/problems/maximum-repeating-substring/



您的支持是我最大的动力
