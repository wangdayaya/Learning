leetcode  17. Letter Combinations of a Phone Number（python）




### 描述

Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/11/09/200px-telephone-keypad2svg.png)


Example 1:

	Input: digits = "23"
	Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

	
Example 2:


	Input: digits = ""
	Output: []

Example 3:


	Input: digits = "2"
	Output: ["a","b","c"]
	


Note:

	0 <= digits.length <= 4
	digits[i] is a digit in the range ['2', '9'].


### 解析


根据题意，给定一个包含 2-9 数字的字符串，返回该数字可以表示的所有可能的字母组合，以任意顺序返回答案都合理。图片中给出了一个拨号键盘，大家可以参考。

其实这道题和刚刚过去的第 292 场周赛中的第三题[ 2266. Count Number of Texts ](https://leetcode.com/contest/weekly-contest-292/problems/count-number-of-texts/)很像，不过周赛中的题目是用到动态规划的，有点难度，大家有兴趣可以看看。


言归正传，这个题是比较简单的，从题目上我们就能看出来这就是一道排列组合的题目，而且限制条件中也定义了 digits 的长度最大为 4 ，也就是说我们直接用多重循环就能解决这道题，我们这里还借用了栈的一些操作。

* 首先我们定义好键盘映射关系 d ，结果列表 result 为空
* 判断如果 digits 的长度为 0 ，直接返回 result 
* 然后我们将 digits[0] 对应的列表先存入 result 中，然后从 digits[1] 开始往后遍历，每次通过对前面的元素出栈，来和新的键盘字母进行组合，并加入 result 中，遍历结束返回 result 即可

时间复杂度为 4 + 4^2 + 4^3 + 4^4 所以最后为 O(4^N)，N 为 digits 的长度，空间复杂度为 O(4^N) 。

### 解答
				

	class Solution(object):
	    def letterCombinations(self, digits):
	        """
	        :type digits: str
	        :rtype: List[str]
	        """
	        d = {
	            '2': list('abc'),
	            '3': list('def'),
	            '4': list('ghi'),
	            '5': list('jkl'),
	            '6': list('mno'),
	            '7': list('pqrs'),
	            '8': list('tuv'),
	            '9': list('wxyz')
	        }
	        result = []
	        if len(digits) == 0:
	            return result
	        for n in d[digits[0]]:
	            result.append(n)
	        for n in digits[1:]:
	            N = len(result)
	            while N>0:
	                a = result.pop(0)
	                for c in d[n]:
	                    result.append(a+c)
	                N -= 1
	        return result
            	      
			
### 运行结果


	Runtime: 21 ms, faster than 62.14% of Python online submissions for Letter Combinations of a Phone Number.
	Memory Usage: 13.5 MB, less than 67.62% of Python online submissions for Letter Combinations of a Phone Number.

### 原题链接

https://leetcode.com/problems/letter-combinations-of-a-phone-number/


您的支持是我最大的动力
