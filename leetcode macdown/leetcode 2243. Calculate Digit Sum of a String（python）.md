leetcode 2243. Calculate Digit Sum of a String （python）

这道题是第 289 场 leetcode 周赛的第一题，难度为 Eazy ，主要考察的是就是对字符串和列表的基本操作。


### 描述


You are given a string s consisting of digits and an integer k.

A round can be completed if the length of s is greater than k. In one round, do the following:

* Divide s into consecutive groups of size k such that the first k characters are in the first group, the next k characters are in the second group, and so on. Note that the size of the last group can be smaller than k.
* Replace each group of s with a string representing the sum of all its digits. For example, "346" is replaced with "13" because 3 + 4 + 6 = 13.
* Merge consecutive groups together to form a new string. If the length of the string is greater than k, repeat from step 1.

Return s after all rounds have been completed.


Example 1:

	Input: s = "11111222223", k = 3
	Output: "135"
	Explanation: 
	- For the first round, we divide s into groups of size 3: "111", "112", "222", and "23".
	  ​​​​​Then we calculate the digit sum of each group: 1 + 1 + 1 = 3, 1 + 1 + 2 = 4, 2 + 2 + 2 = 6, and 2 + 3 = 5. 
	  So, s becomes "3" + "4" + "6" + "5" = "3465" after the first round.
	- For the second round, we divide s into "346" and "5".
	  Then we calculate the digit sum of each group: 3 + 4 + 6 = 13, 5 = 5. 
	  So, s becomes "13" + "5" = "135" after second round. 
	Now, s.length <= k, so we return "135" as the answer.

	
Example 2:


	Input: s = "00000000", k = 3
	Output: "000"
	Explanation: 
	We divide s into "000", "000", and "00".
	Then we calculate the digit sum of each group: 0 + 0 + 0 = 0, 0 + 0 + 0 = 0, and 0 + 0 = 0. 
	s becomes "0" + "0" + "0" = "000", whose length is equal to k, so we return "000".



Note:


	1 <= s.length <= 100
	2 <= k <= 100
	s consists of digits only.

### 解析


根据题意，给定一个由数字组成的字符串 s 和整数 k 。如果 s 的长度大于 k ，则可以完成一轮操作。 在一轮操作中，执行以下操作：

* 将 s 分成大小为 k 的连续数字组，使得前 k 个字符在第一组中，接下来的 k 个字符在第二组中，依此类推。 请注意，最后一组的长度可能小于 k 。
* 将每组 s 替换为表示其所有数字之和的字符串。 例如，“346”被替换为“13”，因为 3 + 4 + 6 = 13。
* 将连续的组合并在一起形成一个新的字符串。 如果字符串的长度大于 k ，则从步骤 1 开始重复新一轮的操作。

最后完成后返回 s 。

这道题考察的就是字符串和列表的基本操作，我们的解题也可以按照题目的要求进行：

* 如果 s 的长度小于等于 k 直接返回 s
* 否则进行 while 循环，将 s 拆分成若干 k 长度的子字符串，然后将每个子字符串进行相加变为新的子字符串放入列表 tmp 中，将 tmp 拼接成字符串赋与 s ，进行下一次循环操作
* 循环结束直接返回 s 即可

时间复杂度为 O(N)，空间复杂度为 O(N)。


### 解答
				
	class Solution(object):
	    def digitSum(self, s, k):
	        """
	        :type s: str
	        :type k: int
	        :rtype: str
	        """
	        if len(s) <=k : return s
	        while len(s)>k:
	            tmp = []
	            for i in range(0, len(s), k):
	                tmp.append(s[i:i+k])
	            for i,t in enumerate(tmp):
	                sum = 0
	                for j in tmp[i]:
	                    sum += int(j)
	                tmp[i] = str(sum)
	            s = ''.join(tmp)
	        return s

            	      
			
### 运行结果

	121 / 121 test cases passed.
	Status: Accepted
	Runtime: 16 ms
	Memory Usage: 13.7 MB


### 原题链接


https://leetcode.com/contest/weekly-contest-289/problems/calculate-digit-sum-of-a-string/


您的支持是我最大的动力
