leetcode 875. Koko Eating Bananas （python）

### 每日经典

《》 ——（）


### 描述

Koko loves to eat bananas. There are n piles of bananas, the i<sub>th</sub> pile has piles[i] bananas. The guards have gone and will come back in h hours.

Koko can decide her bananas-per-hour eating speed of k. Each hour, she chooses some pile of bananas and eats k bananas from that pile. If the pile has less than k bananas, she eats all of them instead and will not eat any more bananas during this hour.

Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.

Return the minimum integer k such that she can eat all the bananas within h hours.



Example 1:

	Input: piles = [3,6,7,11], h = 8
	Output: 4

	
Example 2:
	
	Input: piles = [30,11,23,4,20], h = 5
	Output: 30



Example 3:


	Input: piles = [30,11,23,4,20], h = 6
	Output: 23
	




Note:
	
	1 <= piles.length <= 10^4
	piles.length <= h <= 10^9
	1 <= piles[i] <= 10^9


### 解析

根据题意，Koko 喜欢吃香蕉。 有 n 堆香蕉，第 i 堆有 piles[i] 个香蕉。 守卫已经走了，会在几个小时后回来。

Koko 可以决定她每小时吃香蕉的速度为 k 。 每个小时她都会选择一堆香蕉，然后从那堆香蕉中吃掉 k 根香蕉。 如果这堆香蕉少于 k 个，她会吃掉所有香蕉，并且在这个小时内不会再吃任何香蕉。Koko  喜欢慢慢吃，但还是想在警卫回来之前把香蕉吃完。返回最小整数 k 使得她可以在 h 小时内吃掉所有的香蕉。

这道题用暴力的方法很简单，就是按照题目的思路，假如 piles 的长度等于 h ，说明我们每小时只能以吃 max(piles)  个的速度进行，直接返回 max(piles) 即可。否则我们计算出 result 可能的最小值 sum(piles)//h ，然后不断加一，去测试是否会正好在 p 小时内完成，肯定有一个数字能满足题意，所以返回 result 即可。

但是这种做法肯定会超时，因为看限制条件 piles.length 的长度为 10^4 ，而每个值 piles[i] 的范围最大为 10^9 ，也就是这种解法的最坏时间复杂度为 O(10^4 * 10^9) ，肯定会超时。

### 解答
				


	 class Solution(object):
	    def minEatingSpeed(self, piles, h):
	        """
	        :type piles: List[int]
	        :type h: int
	        :rtype: int
	        """
	        if h == len(piles):
	            return max(piles)
	        
	        result = sum(piles)//h
	        while result > 0:
	            tmp = 0
	            for p in piles:
	                tmp += (p // result + (0 if p % result == 0 else 1))
	            if tmp == h:
	                return result
	            result += 1           	      
			
### 运行结果


	Time Limit Exceeded

### 解析

其实从上面的思路我们可以看出来，遍历一次 piles 这个是肯定的，能够优化的就是找每小时吃香蕉的速度 k ，我们可以使用二分法来进行，因为最慢的速度肯定是每个小时吃 1 个，最快的速度肯定是每小时吃 max(piles) 个香蕉，所以我们用二分法来定位耗时在 h 小时内的最慢吃香蕉数量 k ，整体的代码框架和上面的一样，只需要对二分法使用到的 mn 和 mx 以及 mid 不断进行变化即可。

时间复杂度是 O(nlogm) ，n 是 piles 的长度， m 就是 max(piles) ，这次勉强可以通过，所以这道题的关键考察点就是二分法搜值。


### 解答


	class Solution(object):
	    def minEatingSpeed(self, piles, h):
	        """
	        :type piles: List[int]
	        :type h: int
	        :rtype: int
	        """
	        if h == len(piles):
	            return max(piles)
	        mn = 1
	        mx = max(piles)
	        while mn < mx:
	            mid = (mx + mn) // 2
	            time = 0
	            for p in piles:
	                time += p // mid + (0 if p%mid==0 else 1)
	            if time <= h:
	                mx = mid
	            else:
	                mn = mid + 1
	        return mn


### 运行结果

	Runtime: 592 ms, faster than 41.73% of Python online submissions for Koko Eating Bananas.
	Memory Usage: 14.8 MB, less than 26.21% of Python online submissions for Koko Eating Bananas.

### 原题链接

https://leetcode.com/problems/koko-eating-bananas/



您的支持是我最大的动力
