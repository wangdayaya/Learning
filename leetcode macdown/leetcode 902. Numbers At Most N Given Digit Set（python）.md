leetcode  902. Numbers At Most N Given Digit Set（python）




### 描述


Given an array of digits which is sorted in non-decreasing order. You can write numbers using each digits[i] as many times as we want. For example, if digits = ['1','3','5'], we may write numbers such as '13', '551', and '1351315'.

Return the number of positive integers that can be generated that are less than or equal to a given integer n.

 


Example 1:

	Input: digits = ["1","3","5","7"], n = 100
	Output: 20
	Explanation: 
	The 20 numbers that can be written are:
	1, 3, 5, 7, 11, 13, 15, 17, 31, 33, 35, 37, 51, 53, 55, 57, 71, 73, 75, 77.

	
Example 2:

	Input: digits = ["1","4","9"], n = 1000000000
	Output: 29523
	Explanation: 
	We can write 3 one digit numbers, 9 two digit numbers, 27 three digit numbers,
	81 four digit numbers, 243 five digit numbers, 729 six digit numbers,
	2187 seven digit numbers, 6561 eight digit numbers, and 19683 nine digit numbers.
	In total, this is 29523 integers that can be written using the digits array.


Example 3:


	Input: digits = ["7"], n = 8
	Output: 1


Note:


	1 <= digits.length <= 9
	digits[i].length == 1
	digits[i] is a digit from '1' to '9'.
	All the values in digits are unique.
	digits is sorted in non-decreasing order.
	1 <= n <= 10^9

### 解析

根据题意，给定一个按非降序排序的数组 digits 。 可以根据需要，多次使用每个 digits[i]  编写新的数字。 例如，如果 digits = ['1','3','5'] ，我们可能会写出 '13' 、'551' 和 '1351315' 等这样的数字。返回可以生成的小于或等于给定整数 n 的所有正整数的数量。


这个思路是学习的[灵神的解法](https://www.bilibili.com/video/BV1rS4y1s721?spm_id_from=333.999.0.0&vd_source=66ea1dd09047312f5bc02b99f5652ac6)，
因为有不同的状态进行转换，并且有很多重复的计算，我们使用记忆化的 dfs 来进行解题。按照题意我们可以不断取任意一个数字从前往后拼接来生成新的不同的合法数字，因为 digits 在题目中限制只是 1- 9 的数字，所以可以省略很多边界条件（比如前导 0）的判断。

因为是从左往右的顺序生成数字，所以为了满足生成完整数字时前面位对于后面位的约束，我们定义递归函数 dfs ，表示构造从左往右第 i 位及其之后数位的不同的有效数字方案数，参数有两个如下所示：

- is_limit 表示在构造当前位的时候是否受到了前一位的约束，如果为 True ，那么此位最大为 s[i] ，否则此位可以是 digits 中的任意数字
- is_num 表示 i 前面的数位上是否填了至少一个数字，若为 True 则 i 位之前的所有位置可以形成一个有效数字，否则表示 i 位之前都没有填数字

递归函数内逻辑如下：

- 当 i 等于 s 长度，说明已经到了最后一位，如果此时 is_num 为 True 表示形成了一个有效的数字返回 1 ，否则说明该数字无效返回 0
- 如果第 i 位之前的位上都没有填数字，那么我们可以按照没有前面数字约束的情况下，递归找出最后能够形成的所有有效的数字个数，加到 result 上
- 否则第 i 位之前的位上有数字的时候，我们要根据 is_limit 判断当前位的大小上限 up ，然后在当前位选择比 up 小的数字填充当前位，递归找出最后能够形成的所有有效的数字个数，加到 result 上
- 最后返回 result 即可

时间复杂度为 O(N * len(digits)) ，N 是字符串 n 的长度，空间复杂度为 O(N) 。
### 解答

	class Solution:
	    def atMostNGivenDigitSet(self, digits: List[str], n: int) -> int:
	        s = str(n)
	        @cache
	        def f(i: int, is_limit: bool, is_num: bool) -> int:
	            if i == len(s):
	                return int(is_num)
	            result = 0
	            if not is_num:
	                result += f(i+1, False, False)
	            up = s[i] if is_limit else '9'
	            for num in digits:
	                if num > up:
	                    break
	                result += f(i+1, is_limit and num == up, True)
	            return result
	        return f(0, True, False)
	


### 运行结果

	Runtime: 34 ms, faster than 88.25% of Python3 online submissions for Numbers At Most N Given Digit Set.
	Memory Usage: 13.9 MB, less than 84.34% of Python3 online submissions for Numbers At Most N Given Digit Set.


### 原题链接

https://leetcode.com/problems/numbers-at-most-n-given-digit-set/


您的支持是我最大的动力
