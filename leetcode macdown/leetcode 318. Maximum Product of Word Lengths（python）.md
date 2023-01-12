leetcode  318. Maximum Product of Word Lengths（python）



### 描述


Given a string array words, return the maximum value of length(word[i]) * length(word[j]) where the two words do not share common letters. If no such two words exist, return 0.


Example 1:

	Input: words = ["abcw","baz","foo","bar","xtfn","abcdef"]
	Output: 16
	Explanation: The two words can be "abcw", "xtfn".

	
Example 2:

	Input: words = ["a","ab","abc","d","cd","bcd","abcd"]
	Output: 4
	Explanation: The two words can be "ab", "cd".


Example 3:

	Input: words = ["a","aa","aaa","aaaa"]
	Output: 0
	Explanation: No such pair of words.

	

Note:

	2 <= words.length <= 1000
	1 <= words[i].length <= 1000
	words[i] consists only of lowercase English letters.



### 解析

根据题意，给定一个字符串数组 words，返回 length(word[i]) * length(word[j]) 的最大值，其中两个单词不能有相同的字母。 如果不存在这样的两个词，则返回 0 。

如果按照常规的思路，我们两两进行比对，而假如两个字符串的长度为 L1 和 L2 ，要达到两个字符串中没有相同的字母，那么我们的时间复杂度为 O(N^2 \* (L1 \* L2)) ，这肯定是超时的，我们要保证时间复杂度能够在 O(N^2) 之内，我们肯定要降低两个字符串的比对性能 O(L1 \* L2) 为 O(1) ，根据最近“每日一题”都在考位运算我们就能猜到，这道题也是在考察位运算，我们使用位运算结果来表示字符串，一共 26 个字母对应 26 个二进制位，我们只需要将每个单词表示成二进制即可，这样我们在比对两个字符串所代表的二进制是只要判断 mask[i] & mask[j] 等于 0 即可表示两个字符串没有相同的字符存在，时间复杂度可以降到 O(1) 。

整体的时间复杂度为 O(N^2) ，空间复杂度为 O(N) 。



### 解答
				
	class Solution(object):
	    def maxProduct(self, words):
	        """
	        :type words: List[str]
	        :rtype: int
	        """
	        N = len(words)
	        result = 0
	        mask = [0] * N
	        for i,word in enumerate(words):
	            for j in range(len(word)):
	                mask[i] |= 1 << (ord(word[j]) - ord('a'))
	        for i in range(N):
	            for j in range(i + 1, N):
	                a = words[i]
	                b = words[j]
	                if not mask[i] & mask[j]:
	                    result = max(result, len(a) * len(b))
	        return result

            	      
			
### 运行结果


	Runtime: 499 ms, faster than 67.32% of Python online submissions for Maximum Product of Word Lengths.
	Memory Usage: 14.2 MB, less than 69.91% of Python online submissions for Maximum Product of Word Lengths.

### 原题链接

https://leetcode.com/problems/maximum-product-of-word-lengths/


您的支持是我最大的动力
