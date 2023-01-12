leetcode 2180. Count Integers With Even Digit Sum （python）


### 前言

这是 Weekly Contest 281 的第一题，难度 Eazy ，考查的就是字符串和整数互换以及数字求和等操作，难度相当简单了。话说这次比赛还有奖品，前 20 名分别能拿到 Apple HomePod mini 、Logitech Gaming Mouse 、 LeetCode Backpack、   LeetCode water bottle 、  LeetCode Big  Notebook ，可惜我是没有这个实力了。

### 描述


Given a positive integer num, return the number of positive integers less than or equal to num whose digit sums are even.

The digit sum of a positive integer is the sum of all its digits.


Example 1:


	Input: num = 4
	Output: 2
	Explanation:
	The only integers less than or equal to 4 whose digit sums are even are 2 and 4. 
	
Example 2:


	Input: num = 30
	Output: 14
	Explanation:
	The 14 integers less than or equal to 30 whose digit sums are even are
	2, 4, 6, 8, 11, 13, 15, 17, 19, 20, 22, 24, 26, and 28.






Note:

	1 <= num <= 1000


### 解析

根据题意，给定一个正整数 num，返回小于或等于 num 且 digit sum 为偶数的正整数的个数。题目还给出 digit sum 的定义，就是一个正整数的所有数字的和。

题意相当简单明了，我们再看限制条件，num 的最大值也不过 1000 ，所以比赛中为了缩短思考时间，直接使用无脑暴力求解，从 1 开始遍历到 num 中的每个数字 i ，然后判断每个 i 的  digit sum 是否为偶数，如果是偶数，计数器 reuslt 加一，遍历结束之后直接返回 result 。

时间复杂度为 O(1) ，空间复杂度为 O(1) ，没办法 num 的值太小了。

可能有的朋友觉得这种方法太 low 了，无脑暴力求解没什技术含量，但是在比赛时候排名是按照用时排序的，所以能尽量缩短时间的解法就是最优的，如果你有其他更加便捷的解法当然也是很好的啦。

### 解答
				

	class Solution(object):
	    def countEven(self, num):
	        """
	        :type num: int
	        :rtype: int
	        """
	        result = 0
	        for i in range(1, num+1):
	            L = [int(c) for c in str(i)]
	            if sum(L) % 2 ==0:
	                result += 1
	        return result
            	      
			
### 运行结果


	71 / 71 test cases passed.
	Status: Accepted
	Runtime: 64 ms
	Memory Usage: 13.4 MB

### 原题链接


https://leetcode.com/contest/weekly-contest-281/problems/count-integers-with-even-digit-sum/


您的支持是我最大的动力
