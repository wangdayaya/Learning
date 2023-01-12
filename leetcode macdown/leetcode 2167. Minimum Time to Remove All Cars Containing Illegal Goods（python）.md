leetcode  2167. Minimum Time to Remove All Cars Containing Illegal Goods（python）


### 前言

这是 Weekly Contest 279 比赛的第四题，难度 Hard ，比赛的时候没有什么思路，所以只提交了一次错误的代码，这是看了大佬的解答才豁然开朗，考察的内容是贪心。


### 描述

You are given a 0-indexed binary string s which represents a sequence of train cars. s[i] = '0' denotes that the i<sup>th</sup> car does not contain illegal goods and s[i] = '1' denotes that the i<sup>th</sup> car does contain illegal goods.

As the train conductor, you would like to get rid of all the cars containing illegal goods. You can do any of the following three operations any number of times:

* Remove a train car from the left end (i.e., remove s[0]) which takes 1 unit of time.
* Remove a train car from the right end (i.e., remove s[s.length - 1]) which takes 1 unit of time.
* Remove a train car from anywhere in the sequence which takes 2 units of time.
Return the minimum time to remove all the cars containing illegal goods.

Note that an empty sequence of cars is considered to have no cars containing illegal goods.



Example 1:

	Input: s = "1100101"
	Output: 5
	Explanation: 
	One way to remove all the cars containing illegal goods from the sequence is to
	- remove a car from the left end 2 times. Time taken is 2 * 1 = 2.
	- remove a car from the right end. Time taken is 1.
	- remove the car containing illegal goods found in the middle. Time taken is 2.
	This obtains a total time of 2 + 1 + 2 = 5. 
	
	An alternative way is to
	- remove a car from the left end 2 times. Time taken is 2 * 1 = 2.
	- remove a car from the right end 3 times. Time taken is 3 * 1 = 3.
	This also obtains a total time of 2 + 3 = 5.
	
	5 is the minimum time taken to remove all the cars containing illegal goods. 
	There are no other ways to remove them with less time.

	



Note:


* 1 <= s.length <= 2 * 10^5
* s[i] is either '0' or '1'.

### 解析


根据题意，给一个 0 索引的二进制字符串 s，它代表一系列火车车厢。 s[i] = '0' 表示第 i 节车没有违禁品，s[i] = '1' 表示第 i 节车有违禁品。

作为列车长，您希望去掉所有载有非法物品的车厢，可以多次执行以下三个操作中的任何一个：

* 从左端移除一节火车车厢（即移除 s[0]），这需要 1 个单位时间
* 从右端移除一节火车车厢（即移除 s[s.length - 1]），这需要 1 个单位时间
* 从序列中的任意位置移除一节火车车厢，需要 2 个单位时间

返回清除所有载有非法货物车厢的最短时间。

这是看了大佬的解释才懂，我们可以使用 O(N) 的时间复杂度，从左到右一次遍历 s 即可得到答案。初始化 left 为 0 ，统计从索引 0 到 i 的范围内去掉所有非法物品消耗的时间，right 统计从索引 i+1 末尾范围内去掉所有非法物品消耗的时间。当遍历过程中字符为 1 ，我们要从两种选择中选择较小值：

* 一种方案消耗的时间为 i+1 ，也就是移除非法物品左边所有的车厢的时间
* 一种方案消耗的时间为 left + 2 ，也就是之前移除车厢的时间和移除当前车厢的时间

这个遍历 s 的过程中， right = len(s) - 1 - i 表示的是移除非法物品右边所有车厢所消耗的时间，left + right 就是当前移除所有非法物品的最小时间，不断和 result 比较取较小值赋予 result ，最后返回 result 即可。

### 解答
				
	class Solution(object):
	    def minimumTime(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        left, result, N = 0, len(s), len(s)
	        for i,c in enumerate(s):
	            left = min(left + (c == '1') * 2, i + 1)
	            right = N - 1 - i
	            result = min(result, left + right)
	        return result
            	      
			
### 运行结果


	89 / 89 test cases passed.
	Status: Accepted
	Runtime: 1513 ms
	Memory Usage: 16.2 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-279/problems/minimum-time-to-remove-all-cars-containing-illegal-goods/


您的支持是我最大的动力
