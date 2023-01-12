leetcode  2053. Kth Distinct String in an Array（python）

### 描述


A distinct string is a string that is present only once in an array.

Given an array of strings arr, and an integer k, return the kth distinct string present in arr. If there are fewer than k distinct strings, return an empty string "".

Note that the strings are considered in the order in which they appear in the array.




Example 1:


	Input: arr = ["d","b","c","b","c","a"], k = 2
	Output: "a"
	Explanation:
	The only distinct strings in arr are "d" and "a".
	"d" appears 1st, so it is the 1st distinct string.
	"a" appears 2nd, so it is the 2nd distinct string.
	Since k == 2, "a" is returned. 
	
Example 2:

	Input: arr = ["aaa","aa","a"], k = 1
	Output: "aaa"
	Explanation:
	All strings in arr are distinct, so the 1st string "aaa" is returned.


Example 3:

	Input: arr = ["a","b","a"], k = 3
	Output: ""
	Explanation:
	The only distinct string is "b". Since there are fewer than 3 distinct strings, we return an empty string "".

	

Note:

	
	1 <= k <= arr.length <= 1000
	1 <= arr[i].length <= 5
	arr[i] consists of lowercase English letters.

### 解析

根据题意，给出了一个 distinct string 是在数组中只出现一次的字符串。

给定一个字符串数组 arr 和一个整数 k，返回 arr 中存在的第 k 个  distinct string  。 如果少于 k 个不同的字符串，则返回一个空字符串 ""。

请注意，字符串是按照它们在数组中出现的顺序来考虑的。

思路比较简单，因为要找第 k 个  distinct string  ，所以只需要遍历 arr 中的每个元素，使用内置函数 count 来对每个元素进行计数，如果元素的计数结果为 1 ，则 k 减一，当 k 为 0 的时候说明已经找到了第 k 个 distinct string ，所以直接返回即可，否则就一直按照上述过程遍历找下去，最后如果没有找到直接返回空字符串 “” 即可。

### 解答
				

	class Solution(object):
	    def kthDistinct(self, arr, k):
	        """
	        :type arr: List[str]
	        :type k: int
	        :rtype: str
	        """
	        d = {}
	        for s in arr:
	            if arr.count(s)==1:
	                k-=1
	                if k==0:
	                    return s
	        return ""
	                
	            
            	      
			
### 运行结果

	Runtime: 264 ms, faster than 22.00% of Python online submissions for Kth Distinct String in an Array.
	Memory Usage: 13.9 MB, less than 58.00% of Python online submissions for Kth Distinct String in an Array.



原题链接：https://leetcode.com/problems/kth-distinct-string-in-an-array/



您的支持是我最大的动力
