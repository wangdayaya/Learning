leetcode  605. Can Place Flowers（python）




### 描述


You have a long flowerbed in which some of the plots are planted, and some are not. However, flowers cannot be planted in adjacent plots.

Given an integer array flowerbed containing 0's and 1's, where 0 means empty and 1 means not empty, and an integer n, return if n new flowers can be planted in the flowerbed without violating the no-adjacent-flowers rule.

 


Example 1:
	
	Input: flowerbed = [1,0,0,0,1], n = 1
	Output: true

	
Example 2:

	Input: flowerbed = [1,0,0,0,1], n = 2
	Output: false






Note:

	1 <= flowerbed.length <= 2 * 10^4
	flowerbed[i] is 0 or 1.
	There are no two adjacent flowers in flowerbed.
	0 <= n <= flowerbed.length


### 解析

根据题意，有一个长长的花坛，其中有些地块需要种植，有些地块不需要种植。 但是相邻的地块不能同时种花。

给定一个包含 0 和 1 的整数数组 flowerbed ，其中 0 表示空，1 表示不空，以及一个整数 n ，如果可以在不违反上述种植规则的情况下在花坛中种植 n 朵新花，则返回 True ，否则返回 False 。

根据这块地的种植要求，两朵花不能种在相邻的位置，可能是怕两朵花会因为彼此嫉妒而掐架吧。这道题的解决的思路关键就在这个要求上面，而且我们看了题目要求中 flowerbed.length 最长为 2 * 10^4 ，也就是说只能一次遍历才会不超时地解决问题。这道题虽然是个 Easy 难度，但是通过率只有 32 % ，其实就是人们忽略了很多边界条件，当你细想一下就知道，一共有四种情况：

* 当 flowerbed.length 为 1 的时候，这块地上没有种植花，这时我们可以种植花
* 当 flowerbed.length 大于 1 的情况下，如果索引为 0 的地块是空地，且右边的是空地，这时我们可以种花
* 如果索引为 N-1 的时候，如果这块地是空地，且左边的也是空地，这时我们可以种花
* 如果索引在中间的位置，如果这块地是空地，且左右两边都是空点，这时我们可以种花

我们在遍历 flowerbed 的时候，碰到上面的情况之一，就 n 减一 ，将 flowerbed[i] 变为 1 表示有花种植，同时判断如果 n 为 0 则直接返回 True ，如果遍历结束还没有返回 True ，直接返回 False 即可。


### 解答
				
	class Solution(object):
	    def canPlaceFlowers(self, flowerbed, n):
	        """
	        :type flowerbed: List[int]
	        :type n: int
	        :rtype: bool
	        """
	        if n == 0: return True
	        N = len(flowerbed)
	        for i, f in enumerate(flowerbed):
	            if (i == 0 and i + 1 >= N and flowerbed[i] == 0) or \
	                (i == 0 and i + 1 < N and flowerbed[i + 1] == 0 and flowerbed[i] == 0) or \
	                (i - 1 >= 0 and i + 1 < N and flowerbed[i + 1] == 0 and flowerbed[i - 1] == 0 and flowerbed[i] == 0) or \
	                (i == N - 1 and flowerbed[i - 1] == 0 and flowerbed[i] == 0):
	                n -= 1
	                flowerbed[i] = 1
	                if n == 0:
	                    return True
	        return False
            

            	      
			
### 运行结果


	Runtime: 266 ms, faster than 6.88% of Python online submissions for Can Place Flowers.
	Memory Usage: 13.7 MB, less than 75.18% of Python online submissions for Can Place Flowers.

原题链接：https://leetcode.com/problems/can-place-flowers/



您的支持是我最大的动力
