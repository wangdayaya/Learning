leetcode  1239. Maximum Length of a Concatenated String with Unique Characters（python）




### 描述

You are given an array of strings arr. A string s is formed by the concatenation of a subsequence of arr that has unique characters. Return the maximum possible length of s. A subsequence is an array that can be derived from another array by deleting some or no elements without changing the order of the remaining elements.



Example 1:

	Input: arr = ["un","iq","ue"]
	Output: 4
	Explanation: All the valid concatenations are:
	- ""
	- "un"
	- "iq"
	- "ue"
	- "uniq" ("un" + "iq")
	- "ique" ("iq" + "ue")
	Maximum length is 4.

	
Example 2:

	Input: arr = ["cha","r","act","ers"]
	Output: 6
	Explanation: Possible longest valid concatenations are "chaers" ("cha" + "ers") and "acters" ("act" + "ers").


Example 3:



	Input: arr = ["abcdefghijklmnopqrstuvwxyz"]
	Output: 26
	Explanation: The only string in arr has all 26 characters.

Note:


	1 <= arr.length <= 16
	1 <= arr[i].length <= 26
	arr[i] contains only lowercase English letters.

### 解析

根据题意，给定一个字符串数组 arr ，字符串 s 由具有唯一字符的 arr 子序列的串联而成。返回 s 的最大可能长度。子序列是将一个数组通过删除某些元素或不删除任何元素而不更改其余元素的顺序派生出来的数组。

因为题目中 arr 的长度最大是 16 ，arr[i] 最长为 26 ，所以我们可以直接使用暴力的方法进行解题，我们定义一个，数组 L ，里面存放一个空集合，然后我们开始遍历 arr 中的每个字符串 string ：

* 如果当前字符串中有重复的字符，则直接跳过，不去管他
* 如果当前字符串中没有重复的字符，我们就再遍历 L 中的每个集合 s ，s 挨个和 charSet 进行集合运算，如果有重复的字符则跳过，如果没有重复的字符则将 s 和 charSet 合并起来形成新的集合存入 L 中，不断重复上述操作
* 最后遍历 L 中的各个集合的长度，返回最长的长度即可




### 解答

	class Solution(object):
	    def maxLength(self, arr):
	        """
	        :type arr: List[str]
	        :rtype: int
	        """
	        L = [set()]
	        for string in arr:
	            if len(set(string)) < len(string): continue
	            charSet = set(string)
	            for s in L:
	                if s & charSet: continue
	                L.append(s | charSet)
	        return max(len(s) for s in L)

### 运行结果

	Runtime: 117 ms, faster than 84.68% of Python online submissions for Maximum Length of a Concatenated String with Unique Characters.
	Memory Usage: 52 MB, less than 9.01% of Python online submissions for Maximum Length of a Concatenated String with Unique Characters.


### 原题链接

https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/


您的支持是我最大的动力
