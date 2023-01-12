leetcode  2251. Number of Flowers in Full Bloom（python）




### 描述

You are given a 0-indexed 2D integer array flowers, where flowers[i] = [starti, endi] means the ith flower will be in full bloom from starti to endi (inclusive). You are also given a 0-indexed integer array persons of size n, where persons[i] is the time that the ith person will arrive to see the flowers.

Return an integer array answer of size n, where answer[i] is the number of flowers that are in full bloom when the ith person arrives.



Example 1:

![](https://assets.leetcode.com/uploads/2022/03/02/ex1new.jpg)

	Input: flowers = [[1,6],[3,7],[9,12],[4,13]], persons = [2,3,7,11]
	Output: [1,2,2,2]
	Explanation: The figure above shows the times when the flowers are in full bloom and when the people arrive.
	For each person, we return the number of flowers in full bloom during their arrival.

	
Example 2:

![](https://assets.leetcode.com/uploads/2022/03/02/ex2new.jpg)

	Input: flowers = [[1,10],[3,3]], persons = [3,3,2]
	Output: [2,2,1]
	Explanation: The figure above shows the times when the flowers are in full bloom and when the people arrive.
	For each person, we return the number of flowers in full bloom during their arrival.







Note:

1 <= flowers.length <= 5 * 10^4
flowers[i].length == 2
1 <= starti <= endi <= 10^9
1 <= persons.length <= 5 * 10^4
1 <= persons[i] <= 10^9


### 解析


根据题意，给定一个 0 索引的 2D 整数数组 flowers ，其中 flowers[i] = [starti, endi] 表示第 i 朵花将从 starti 到 endi 盛开。 给出一个大小为 n 的 0 索引整数数组 people，其中 persons[i] 是第 i 个人看花的时间。

返回一个大小为 n 的整数数组 answer，其中 answer[i] 是第 i 个人到达时盛开的花朵数。

这个题本身是不难的，难是难在了常规的解法会超时，因为 flowers 的长度最大为 5 * 10^4 ，persons 的长度最大为 5 * 10^4 ，所以双重循环肯定是超时了。

这里我们将所有花的开花时间都保存在一个列表 A 中并进行升序排序，将所有花的结束时间都保存在一个列表 B 中并进行升序排序，然后我们遍历每个人的看花时间 p  ，然后我们只需要从 A 中找出在 p 时刻已经开花的花朵数量 i ，从 B 中找出在 p 时刻已经谢花的花朵数量 j ，i-j 就是 p 时刻正在开花的数量，加入结果中即可。遍历结束返回 result 即可。

因为每次循环都有二分查找操作，所以时间复杂度为 O(NlogN)，空间复杂度为 O(N) 。

### 解答
				
	class Solution(object):
	    def fullBloomFlowers(self, flowers, persons):
	        """
	        :type flowers: List[List[int]]
	        :type persons: List[int]
	        :rtype: List[int]
	        """
	        A = sorted([x for x, y in flowers])  
	        B = sorted([y for x, y in flowers])  
	        result = []
	        for p in persons:
	            i = bisect.bisect_right(A, p)  
	            j = bisect.bisect_left(B, p)  
	            result.append(i - j) 
	        return result

            	      
			
### 运行结果

	52 / 52 test cases passed.
	Status: Accepted
	Runtime: 961 ms
	Memory Usage: 44.7 MB


### 原题链接


https://leetcode.com/contest/weekly-contest-290/problems/number-of-flowers-in-full-bloom/



您的支持是我最大的动力
