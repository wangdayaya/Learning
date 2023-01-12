leetcode  2375. Construct Smallest Number From DI String （python）




### 描述

You are given a 0-indexed string pattern of length n consisting of the characters 'I' meaning increasing and 'D' meaning decreasing. A 0-indexed string num of length n + 1 is created using the following conditions:

* num consists of the digits '1' to '9', where each digit is used at most once.
* If pattern[i] == 'I', then num[i] < num[i + 1].
* If pattern[i] == 'D', then num[i] > num[i + 1].

Return the lexicographically smallest possible string num that meets the conditions.



Example 1:

	Input: pattern = "IIIDIDDD"
	Output: "123549876"
	Explanation:
	At indices 0, 1, 2, and 4 we must have that num[i] < num[i+1].
	At indices 3, 5, 6, and 7 we must have that num[i] > num[i+1].
	Some possible values of num are "245639871", "135749862", and "123849765".
	It can be proven that "123549876" is the smallest possible num that meets the conditions.
	Note that "123414321" is not possible because the digit '1' is used more than once.

	
Example 2:

	Input: pattern = "DDD"
	Output: "4321"
	Explanation:
	Some possible values of num are "9876", "7321", and "8742".
	It can be proven that "4321" is the smallest possible num that meets the conditions.




Note:


	1 <= pattern.length <= 8
	pattern consists of only the letters 'I' and 'D'.

### 解析

根据题意，给定一个长度为 n 的 0 索引字符串模式，其中包含字符“I”表示增加，“D”表示减少。 使用以下条件创建长度为 n + 1 的 0 索引字符串 num：

* num 由数字 '1' 到 '9' 组成，其中每个数字最多使用一次
* 如果模式[i] == 'I'，则 num[i] < num[i + 1]
* 如果模式[i] == 'D'，则 num[i] > num[i + 1]

返回满足条件的字典序最小的可能字符串 num。

这道题考查的就是贪心算法，我们要想要最后的字符串是字典序最小的数字，我们就要尽可能的把小的数字放到前面，这是基本的原则，我们初始化一个 L ，里面按升序保存着 1 - 9 这些数字，这时候我们不断从前往后遍历 pattern 中的每个字母，碰到 I 或者 D 需要进行不同的处理：

* 当我们碰到 I 的时候，我们只需要将 L.pop(0) 这个数字加到 result 后面即可，因为 L 最前面的数字就是我们在剩下没用的数字里面最小的数字了，符合题目的基本原则
* 当我们碰到 D 的时候，我们需要知道有几个挨着的 D ，如例子一 “IIIDIDDD” 字符串遍历到第四个字母 D 的时候，我们计算出此时其紧挨的 D 只有它一个，所以我们为了保证此时是降序，且尽量用小的数字，所以我们要把 L.pop(cnt) 拼接到 result 后面，cnt 就是此时连续 D 的个数，此时索引为 cnt 的元素就是剩下能用的最小的并且满足降序的数字。

遍历 pattern 结束，最后将 L[0] 这个此时可用的最小数字拼接到 result 即可。最后返回 result 。时间复杂度为 O(N^2) ，空间复杂度为 O(N)。

### 解答
	class Solution(object):
	    def smallestNumber(self, pattern):
	        """
	        :type pattern: str
	        :rtype: str
	        """
	        N = len(pattern)
	        L = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
	        result = ''
	        for i, c in enumerate(pattern):
	            if c == 'I':
	                result += L.pop(0)
	            elif c == 'D':
	                j = i
	                cnt = 1
	                while j+1 < N and pattern[j+1] == 'D':
	                    cnt += 1
	                    j += 1
	                result += L.pop(cnt)
	        result += L.pop(0)
	        return result


### 运行结果


	
	104 / 104 test cases passed.
	Status: Accepted
	Runtime: 28 ms
	Memory Usage: 13.6 MB
### 原题链接

	https://leetcode.com/contest/weekly-contest-306/problems/construct-smallest-number-from-di-string/



您的支持是我最大的动力
