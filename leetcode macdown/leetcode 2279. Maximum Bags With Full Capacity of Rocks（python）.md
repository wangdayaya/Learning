leetcode 2279. Maximum Bags With Full Capacity of Rocks （python）



### 描述

You have n bags numbered from 0 to n - 1. You are given two 0-indexed integer arrays capacity and rocks. The ith bag can hold a maximum of capacity[i] rocks and currently contains rocks[i] rocks. You are also given an integer additionalRocks, the number of additional rocks you can place in any of the bags.

Return the maximum number of bags that could have full capacity after placing the additional rocks in some bags.

 



Example 1:

	Input: capacity = [2,3,4,5], rocks = [1,2,4,4], additionalRocks = 2
	Output: 3
	Explanation:
	Place 1 rock in bag 0 and 1 rock in bag 1.
	The number of rocks in each bag are now [2,3,4,4].
	Bags 0, 1, and 2 have full capacity.
	There are 3 bags at full capacity, so we return 3.
	It can be shown that it is not possible to have more than 3 bags at full capacity.
	Note that there may be other ways of placing the rocks that result in an answer of 3.

	
Example 2:


	Input: capacity = [10,2,2], rocks = [2,2,0], additionalRocks = 100
	Output: 3
	Explanation:
	Place 8 rocks in bag 0 and 2 rocks in bag 2.
	The number of rocks in each bag are now [10,2,2].
	Bags 0, 1, and 2 have full capacity.
	There are 3 bags at full capacity, so we return 3.
	It can be shown that it is not possible to have more than 3 bags at full capacity.
	Note that we did not use all of the additional rocks.



Note:

	n == capacity.length == rocks.length
	1 <= n <= 5 * 10^4
	1 <= capacity[i] <= 10^9
	0 <= rocks[i] <= capacity[i]
	1 <= additionalRocks <= 10^9



### 解析


根据题意，有 n 个袋子，编号从 0 到 n - 1 。有两个索引为 0 的整数数组 capacity 和 rocks 。 第 i 个袋子可以容纳最大 capacity[i] 块岩石，但是目前已经有了 rocks[i]  块岩石。 另外还给出一个整数 additionalRocks ，即可以在任何袋子中放置的附加岩石的数量。在将额外的石头放入任意袋子后，返回可以满载的袋子的最大数量。

这道题一看就是考察贪心的算法，用我们最朴素的想法就能解题，某个袋子的最大容量减去已经有的容量，就是这个袋子现在还空余的容量，我们手里有额外的岩石可以用，那肯定是先装空余容量较小的袋子，这样才能多装满一些袋子啊，一边装满袋子一边统计个数，最后返回统计好的装满的袋子数量即可，这就是解题的整体思路，应该是比较容易的。


时间复杂度为 O(N)，空间复杂度为 O(N) 。

### 解答
				

	class Solution(object):
	    def maximumBags(self, capacity, rocks, additionalRocks):
	        """
	        :type capacity: List[int]
	        :type rocks: List[int]
	        :type additionalRocks: int
	        :rtype: int
	        """
	        N = len(rocks)
	        remaining = [capacity[i]-rocks[i] for i in range(N)]
	        remaining.sort()
	        result = 0
	        for r in remaining:
	            if r == 0:
	                 result += 1
	            elif additionalRocks >= r:
	                additionalRocks -= r
	                result += 1
	            else:
	                break
	        return result
                
            	      
			
### 运行结果

	79 / 79 test cases passed.
	Status: Accepted
	Runtime: 754 ms
	Memory Usage: 21.3 MB


### 原题链接

https://leetcode.com/contest/weekly-contest-294/problems/maximum-bags-with-full-capacity-of-rocks/


您的支持是我最大的动力
