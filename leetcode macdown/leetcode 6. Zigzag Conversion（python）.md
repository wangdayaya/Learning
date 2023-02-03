leetcode  6. Zigzag Conversion（python）




### 描述

The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

	P   A   H   N
	A P L S I I G
	Y   I   R

And then read line by line: "PAHNAPLSIIGYIR"，Write the code that will take a string and make this conversion given a number of rows:

	string convert(string s, int numRows);



Example 1:

	Input: s = "PAYPALISHIRING", numRows = 3
	Output: "PAHNAPLSIIGYIR"

	
Example 2:

	Input: s = "PAYPALISHIRING", numRows = 4
	Output: "PINALSIGYAHRPI"
	Explanation:
	P     I    N
	A   L S  I G
	Y A   H R
	P     I


Example 3:


	Input: s = "A", numRows = 1
	Output: "A"


Note:


	1 <= s.length <= 1000
	s consists of English letters (lower-case and upper-case), ',' and '.'.
	1 <= numRows <= 1000

### 解析

根据题意，字符串“PAYPALISHIRING”中的每个字符以 Z 锯齿形模式写在给定数量的行上，如下所示，将“PAYPALISHIRING”按照 3 行进行排列：

	P   A   H   N
	A P L S I I G
	Y   I   R

然后逐行阅读：“PAHNAPLSIIGYIR”，编写将接受一个字符串并在给定行数的情况下进行此转换的代码：

	string convert(string s, int numRows);


其实这道题就是考察通过模拟过程来解题，我们按照最朴素的想法，给定了 numRows ，那么我们最后的结果肯定只有 numRows 行，假设我们现在定义了一个列表  result ，里面有 numRows 个空字符串，然后我们可以将遍历 s  中的每个字符：

* 开始的时候，每个字符肯定是按照行索引 i 从上到下加入到对应索引的 result[i] 字符串中
* 到最底下的一行之后又会从下往上按照行索引 i 将字符加入到对应索引的  result[i] 字符串中
* 不断重复这个过程，通过模拟 Z 字型的字符排列走向，将每个字符加到对应的 result 中的字符串，最后将 result 拼接成字符串即可得到结果。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。



### 解答

	class Solution(object):
	    def convert(self, s, numRows):
	        """
	        :type s: str
	        :type numRows: int
	        :rtype: str
	        """
	        if numRows < 2: return s
	        result = ["" for _ in range(numRows)]
	        i, flag = 0, -1
	        for c in s:
	            result[i] += c
	            if i == 0 or i == numRows - 1:
	                flag = -flag
	            i += flag
	        return "".join(result)

### 运行结果
	Runtime 47 ms ，Beats 73.87%
	Memory 13.9 MB ，Beats 13.18%


### 原题链接

https://leetcode.com/problems/zigzag-conversion/



您的支持是我最大的动力
