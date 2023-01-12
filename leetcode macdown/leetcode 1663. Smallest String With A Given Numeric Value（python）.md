leetcode  1663. Smallest String With A Given Numeric Value（python）




### 描述


The numeric value of a lowercase character is defined as its position (1-indexed) in the alphabet, so the numeric value of a is 1, the numeric value of b is 2, the numeric value of c is 3, and so on.

The numeric value of a string consisting of lowercase characters is defined as the sum of its characters' numeric values. For example, the numeric value of the string "abe" is equal to 1 + 2 + 5 = 8.

You are given two integers n and k. Return the lexicographically smallest string with length equal to n and numeric value equal to k.

Note that a string x is lexicographically smaller than string y if x comes before y in dictionary order, that is, either x is a prefix of y, or if i is the first position such that x[i] != y[i], then x[i] comes before y[i] in alphabetic order.

 


Example 1:

	Input: n = 3, k = 27
	Output: "aay"
	Explanation: The numeric value of the string is 1 + 1 + 25 = 27, and it is the smallest string with such a value and length equal to 3.

	




Note:

	1 <= n <= 10^5
	n <= k <= 26 * n


### 解析


这道题是结合了字典顺序、字符串拼接、贪心等多个方面综合考察同学们的解题能力，比较有意思。

根据题意，我们首先要知道对于小写字母的数值定义是从 1 开始的，所以 a 的数值为 1 ，b 的数值为 2 ，c 的数值为 3 ，以此类推。小写字母字符组成的字符串的数值定义为其所有字母字符的数值之和。 例如，字符串 “abe” 的数值等于 1 + 2 + 5 = 8 。

了解了以上题目给出的定义， 给定两个整数 n 和 k 。 返回长度等于 n 且字符串数值等于 k 的字典顺序最小字符串。字典顺序其实就是尽量把字母数值较小的字母放到前面，字母数值较大的字母放到后面。

这道题其实就是用贪心的想法来解题，我们就是根据题意，尽量把数值比较小的字母放到前面，数值比较大的字母放到后面。但是我们要知道 k 的数值是固定的，所以我们可以将思路反转，我们从后面放前面进行放置字母字符：

* 定义结果 result 为空字符串，从 n 遍历到 0 （不含） ，一共 n 轮遍历
* 最后一个字母最大肯定是字母数值为 26 的字母 ，也就是 z ，但是我们要保证在放置了 z 之后， k 剩下的数值至少可以保证在前面的字母都可以放置最小数值字母 a ，所以我们要找 tmp=min(26, k-i+1) ，将 tmp 对应的字母追加到 result 之后（如例子一，如果我们最后直接放置了 z ，那么第二位是 a ，第一位将会没有字母，这不满足长度为 n 的条件）
* 然后 k 减去 tmp ，然后再去进行后面的遍历，操作如上
* 遍历结束，因为我们是从后向前的顺序安置字母，所以最后将 result 进行反转即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。
### 解答
				

	class Solution(object):
	    def getSmallestString(self, n, k):
	        result = ''
	        for i in range(n, 0, -1):
	            tmp = min(26, k-i+1)
	            result += chr(ord('a') + tmp - 1)
	            k -= tmp
	        return result[::-1]
            	      
			
### 运行结果


	Runtime: 1156 ms, faster than 61.54% of Python online submissions for Smallest String With A Given Numeric Value.
	Memory Usage: 18 MB, less than 53.85% of Python online submissions for Smallest String With A Given Numeric Value.


### 原题链接



https://leetcode.com/problems/smallest-string-with-a-given-numeric-value/


您的支持是我最大的动力
