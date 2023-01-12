leetcode  2273. Find Resultant Array After Removing Anagrams（python）




### 描述

You are given a 0-indexed string array words, where words[i] consists of lowercase English letters.

In one operation, select any index i such that 0 < i < words.length and words[i - 1] and words[i] are anagrams, and delete words[i] from words. Keep performing this operation as long as you can select an index that satisfies the conditions.

Return words after performing all operations. It can be shown that selecting the indices for each operation in any arbitrary order will lead to the same result.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase using all the original letters exactly once. For example, "dacb" is an anagram of "abdc".



Example 1:


	Input: words = ["abba","baba","bbaa","cd","cd"]
	Output: ["abba","cd"]
	Explanation:
	One of the ways we can obtain the resultant array is by using the following operations:
	- Since words[2] = "bbaa" and words[1] = "baba" are anagrams, we choose index 2 and delete words[2].
	  Now words = ["abba","baba","cd","cd"].
	- Since words[1] = "baba" and words[0] = "abba" are anagrams, we choose index 1 and delete words[1].
	  Now words = ["abba","cd","cd"].
	- Since words[2] = "cd" and words[1] = "cd" are anagrams, we choose index 2 and delete words[2].
	  Now words = ["abba","cd"].
	We can no longer perform any operations, so ["abba","cd"] is the final answer.
	
Example 2:


	Input: words = ["a","b","c","d","e"]
	Output: ["a","b","c","d","e"]
	Explanation:
	No two adjacent strings in words are anagrams of each other, so no operations are performed.




Note:


	1 <= words.length <= 100
	1 <= words[i].length <= 10
	words[i] consists of lowercase English letters.

### 解析

根据题意，给定一个索引为 0 的字符串数组 words，其中 words[i] 由小写英文字母组成。在一个操作中，选择任何索引 i 使得 0 < i < words.length 并且 words[i-1] 和 words[i] 是 anagrams ，将  words[i] 从 words 中删除。 只要可以选择满足条件的索引，就一直执行这个操作。anagrams 表示相同的字母集合通过不同顺序展现出的不同字符串。 例如，“dacb”是“abdc” 的 anagrams。

这道题其实就是考察一个字符串的计数器使用方法，我们既然用 python ，肯定要使用  collections.Counter 这个函数，我们只要根据题意介绍的 anagrams 定义，开始在 result 中放入第一个字符串，然后从第二个字符串开始找，只要新来的字符串 a 和 result 最后一个字符串 b 两个字符串的计数器 a 和 b 不同的情况下，才会将新的字符串加入到结果中，遍历结束之后。

时间复杂度为 O(N) ，空间复杂度为 O(N)。

### 解答
				

	class Solution(object):
	    def removeAnagrams(self, words):
	        """
	        :type words: List[str]
	        :rtype: List[str]
	        """
	        result = [words[0]]
	        for i in range(1, len(words)):
	            a = collections.Counter(words[i])
	            b = collections.Counter(result[-1])
	            if a!=b:
	                result.append(words[i])
	        return result
            	      
			
### 运行结果


	201 / 201 test cases passed.
	Status: Accepted
	Runtime: 103 ms
	Memory Usage: 13.6 MB

### 原题链接


https://leetcode.com/contest/weekly-contest-293/problems/find-resultant-array-after-removing-anagrams/


您的支持是我最大的动力
