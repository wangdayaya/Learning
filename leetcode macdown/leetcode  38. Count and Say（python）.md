leetcode  38. Count and Say（python）




### 描述


The count-and-say sequence is a sequence of digit strings defined by the recursive formula:

* countAndSay(1) = "1"
* countAndSay(n) is the way you would "say" the digit string from countAndSay(n-1), which is then converted into a different digit string.

To determine how you "say" a digit string, split it into the minimal number of substrings such that each substring contains exactly one unique digit. Then for each substring, say the number of digits, then say the digit. Finally, concatenate every said digit. For example, the saying and conversion for digit string "3322251":

![](https://assets.leetcode.com/uploads/2020/10/23/countandsay.jpg)


Given a positive integer n, return the n<sup>th</sup> term of the count-and-say sequence.




Example 1:

	Input: n = 1
	Output: "1"
	Explanation: This is the base case.

	
Example 2:

	Input: n = 4
	Output: "1211"
	Explanation:
	countAndSay(1) = "1"
	countAndSay(2) = say "1" = one 1 = "11"
	countAndSay(3) = say "11" = two 1's = "21"
	countAndSay(4) = say "21" = one 2 + one 1 = "12" + "11" = "1211"




Note:

	1 <= n <= 30


### 解析

根据题意， count-and-say 序列是由递归公式定义的数字字符串序列：

* countAndSay(1) = "1"
* countAndSay(n) 是从  countAndSay(n-1)  转换出来的新的数字字符串。
* 转换规则要通过先计数再说的方式，比如对“3322251”进行转换，我们先说出来，有两个 3 ，三个 2 ，一个 5 ，1 个 1，然后将每个数字的个数和本身从前往后拼接出来就是转换的结果 “23321511”

给定一个正整数 n ，返回第 n 个通过 count-and-say  方式转换后的字符串结果。

其实这道题很简单，限制条件中给出了 n 的最大值为 30 ，所以我们可以采用模拟法，直接进行 n 的模拟即可得到最后的结果，每次模拟的过程就是从左到右遍历，计算相同的相邻字符的个数并和其自身进行字符串拼接，这样遍历结束得到的结果就是一次 countAndSay 的结果，经过 n 次这样的操作得到的 result 就是最后的结果。

时间复杂度为 O(N\*M) ，N 为给定的 n ，M 为转换过程生成的最长的字符串长度 ，空间复杂度为 O(1) 。

其实还可以用打表法进行解题，因为 n 的最大值为 30 ，所以我们可以提前先将这 30 种结果计算出来放到一个列表中，最后只需要根据 n 来进行索引即可，这种很简单就不写代码了。这种方法的时间复杂度为 O(1) ，空间复杂度为 O(N)。



### 解答

	class Solution(object):
	    def countAndSay(self, n):
	        """
	        :type n: int
	        :rtype: str
	        """
	        result = '1'
	        for _ in range(n-1):
	            i = 0
	            pre = 0
	            cur = ''
	            while i < len(result):
	                while i< len(result) and result[i] == result[pre]:
	                    i += 1
	                cur += str(i-pre) + result[pre]
	                pre = i
	            result = cur
	        return result
	 

### 运行结果

	Runtime: 38 ms, faster than 85.50% of Python online submissions for Count and Say.
	Memory Usage: 13.3 MB, less than 96.80% of Python online submissions for Count and Say.


### 原题链接

	https://leetcode.com/problems/count-and-say/


您的支持是我最大的动力
