leetcode 2194. Cells in a Range on an Excel Sheet （python）


### 前言

这是 Weekly Contest 283 第一题，考察的就是基本的字符串拼接，难度 Eazy ，很简单就能做出来。

### 描述


A cell (r, c) of an excel sheet is represented as a string "<col><row>" where:

* <col> denotes the column number c of the cell. It is represented by alphabetical letters.
	
	For example, the 1st column is denoted by 'A', the 2nd by 'B', the 3rd by 'C', and so on.

* <row> is the row number r of the cell. The rth row is represented by the integer r.
You are given a string s in the format "<col1><row1>:<col2><row2>", where <col1> represents the column c1, <row1> represents the row r1, <col2> represents the column c2, and <row2> represents the row r2, such that r1 <= r2 and c1 <= c2.

Return the list of cells (x, y) such that r1 <= x <= r2 and c1 <= y <= c2. The cells should be represented as strings in the format mentioned above and be sorted in non-decreasing order first by columns and then by rows.


Example 1:


![](https://assets.leetcode.com/uploads/2022/02/08/ex1drawio.png)	

	Input: s = "K1:L2"
	Output: ["K1","K2","L1","L2"]
	Explanation:
	The above diagram shows the cells which should be present in the list.
	The red arrows denote the order in which the cells should be presented.



Note:

	s.length == 5
	'A' <= s[0] <= s[3] <= 'Z'
	'1' <= s[1] <= s[4] <= '9'
	s consists of uppercase English letters, digits and ':'.


### 解析


这道题虽然是第一题，但是篇幅很长，其实看起来篇幅长，结合理解起来很简单，因为我们平时用 Excel 表格经常就会碰到这种现象。

根据题意，给出的 s 是一个长度为 5 字符串，其中用冒号分割成两部分，第一部分是起始点的位置，第二部分是终止点的位置，神奇的就是不管是起始点还是终止点都是字母+数字表示的，题目要让我们返回在起始点和终止点形成的矩形范围内，如果第一列在范围内，把第一列放到结果列表中，然后如果有第二列则把第二列放入结果列表中，以此类推，最后返回结果列表即可。说白了就一句话，按列返回矩形中每个位置。

这么一来思路就清楚了，字母表示的是列，数字表示的是行，我们先根据字母的范围遍历列，然后根据数字的范围遍历行，每个位置我们将列字母和行数字拼接起来放入结果列表 result 中即可，遍历结束返回 result 。

因为题目中限制了字母和数字的范围，所以时间复杂度基本是常数 O(1) ，空间复杂度为 O(1) 。

### 解答
				

	class Solution(object):
	    def cellsInRange(self, s):
	        """
	        :type s: str
	        :rtype: List[str]
	        """
	        result = []
	        for c in range(ord(s[0]), ord(s[3])+1):
	            for n in range(int(s[1]), int(s[4])+1):
	                result.append(chr(c)+str(n))
	        return result
            	      
			
### 运行结果


	251 / 251 test cases passed.
	Status: Accepted
	Runtime: 32 ms
	Memory Usage: 13.8 MB

### 原题链接


https://leetcode.com/contest/weekly-contest-283/problems/cells-in-a-range-on-an-excel-sheet/


您的支持是我最大的动力
