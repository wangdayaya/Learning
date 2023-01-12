leetcode  1642. Furthest Building You Can Reach（python）




### 描述

You are given an integer array heights representing the heights of buildings, some bricks, and some ladders.

You start your journey from building 0 and move to the next building by possibly using bricks or ladders.

While moving from building i to building i+1 (0-indexed),

* If the current building's height is greater than or equal to the next building's height, you do not need a ladder or bricks.
* If the current building's height is less than the next building's height, you can either use one ladder or (h[i+1] - h[i]) bricks.

Return the furthest building index (0-indexed) you can reach if you use the given ladders and bricks optimally.



Example 1:


![](https://assets.leetcode.com/uploads/2020/10/27/q4.gif)

	Input: heights = [4,2,7,6,9,14,12], bricks = 5, ladders = 1
	Output: 4
	Explanation: Starting at building 0, you can follow these steps:
	- Go to building 1 without using ladders nor bricks since 4 >= 2.
	- Go to building 2 using 5 bricks. You must use either bricks or ladders because 2 < 7.
	- Go to building 3 without using ladders nor bricks since 7 >= 6.
	- Go to building 4 using your only ladder. You must use either bricks or ladders because 6 < 9.
	It is impossible to go beyond building 4 because you do not have any more bricks or ladders.
	
Example 2:

	Input: heights = [4,12,2,7,3,18,20,3,19], bricks = 10, ladders = 2
	Output: 7


Example 3:

	Input: heights = [14,3,19,3], bricks = 17, ladders = 0
	Output: 3

	




Note:

	1 <= heights.length <= 10^5
	1 <= heights[i] <= 10^6
	0 <= bricks <= 10^9
	0 <= ladders <= heights.length


### 解析

根据题意，给定一个整数数组 heights，表示建筑物的高度，还有一些砖块和一些梯子。从 0 号楼开始走，并可能使用砖块或梯子移动到下一栋楼。从建筑 i 移动到建筑 i+1（0 索引）时，

* 如果当前建筑物的高度大于或等于下一建筑物的高度，则不需要梯子或砖块。
* 如果当前建筑物的高度小于下一建筑物的高度，您可以使用一个梯子或 (h[i+1] - h[i]) 块砖。

如果以最佳方式使用给定的梯子和砖块，则返回可以达到的最远的建筑物索引（0 开始索引）。

这道题考查的就是贪心思想加堆排序，我们在楼顶进行移动的时候，只需要在爬高处的时候需要进行用砖或者梯子的操作，因为梯子是无限高的，所以根据贪心的思想，肯定是先尽量用砖，但是当砖用完的时候，我们需要注意的是，此时假如我们手里没有了砖且只有一个梯子，用完之后再碰到高楼肯定就停止了，但如果我们用梯子把之前用砖最多的地方的砖都替换出来，我们手里就又会多出来很多砖，再遇到能力范围内的高楼，我们还能继续前进下去，这样操作可以使得爬楼过程可能走的更远。这里不断找之前用砖最多的思路就是用到了大根堆。

时间复杂度为 O(NlogN) ，空间复杂度为 O(N)。


### 解答
				
	class Solution(object):
	    def furthestBuilding(self, heights, bricks, ladders):
	        """
	        :type heights: List[int]
	        :type bricks: int
	        :type ladders: int
	        :rtype: int
	        """
	        L = []
	        for i in range(1, len(heights)):
	            d = heights[i] - heights[i-1]
	            if d > 0:
	                heapq.heappush(L, -d)
	                bricks -= d
	                if bricks < 0:
	                    if ladders > 0:
	                        ladders -= 1
	                        bricks += -heapq.heappop(L)
	                    else:
	                        return i-1
	        return len(heights)-1

            	      
			
### 运行结果


	Runtime: 1035 ms, faster than 34.48% of Python online submissions for Furthest Building You Can Reach.
	Memory Usage: 24.3 MB, less than 22.41% of Python online submissions for Furthest Building You Can Reach.

### 原题链接

https://leetcode.com/problems/furthest-building-you-can-reach/


您的支持是我最大的动力
