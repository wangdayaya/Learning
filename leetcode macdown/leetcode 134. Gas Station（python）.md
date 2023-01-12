leetcode  134. Gas Station（python）




### 描述

There are n gas stations along a circular route, where the amount of gas at the i<sub>th</sub> station is gas[i].

You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the i<sub>th</sub> station to its next (i + 1)<sub>th</sub> station. You begin the journey with an empty tank at one of the gas stations.

Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1. If there exists a solution, it is guaranteed to be unique



Example 1:

	Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
	Output: 3
	Explanation:
	Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
	Travel to station 4. Your tank = 4 - 1 + 5 = 8
	Travel to station 0. Your tank = 8 - 2 + 1 = 7
	Travel to station 1. Your tank = 7 - 3 + 2 = 6
	Travel to station 2. Your tank = 6 - 4 + 3 = 5
	Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
	Therefore, return 3 as the starting index.

	
Example 2:

	Input: gas = [2,3,4], cost = [3,4,3]
	Output: -1
	Explanation:
	You can't start at station 0 or 1, as there is not enough gas to travel to the next station.
	Let's start at station 2 and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
	Travel to station 0. Your tank = 4 - 3 + 2 = 3
	Travel to station 1. Your tank = 3 - 3 + 3 = 3
	You cannot travel back to station 2, as it requires 4 unit of gas but you only have 3.
	Therefore, you can't travel around the circuit once no matter where you start.




Note:

	gas.length == n
	cost.length == n
	1 <= n <= 10^5
	0 <= gas[i], cost[i] <= 10^4


### 解析

根据题意，沿环形路线有 n 个加油站，第 i 个加油站的加油量为 gas[i] 。假设有一辆油箱无限的汽车，从第 i 个站到下一个 (i + 1) 站需要 cost[i]  汽油。 我们从其中一个加油站的空油箱开始旅程。给定两个整数数组 gas 和 cost，如果可以沿顺时针方向绕路一圈，则返回起始加油站的索引，否则返回 -1 。 

这种题一看就知道用暴力是最无脑解决的题目，但是一看限制条件列表长度最大为 10^5 ，那基本就没法用暴力，因为这道题目很明显得两层循环，第一层循环先是遍历每个索引，然后第二层循环就是去判断是不是汽车在经过了若干座加油站是否能顺利返回原处。这么大的计算量最坏的情况肯定是  O(n^2) 超时的结果，但是我们可以先写出来这个代码思路。

其实很多题都是这种解决方法，我们先用最朴素的思路去找规律解题，如果限制条件很宽松，那基本上就可以通过，如果限制条件很严格，那肯定会超时，但是这不妨碍我们的思路，很多优化的细节就是在暴力解法的基础上对关键的耗时步骤进行优化的。


### 解答
				

	class Solution(object):
	    def canCompleteCircuit(self, gas, cost):
	        """
	        :type gas: List[int]
	        :type cost: List[int]
	        :rtype: int
	        """
	        for i in range(len(gas)):
	            if self.isCan(i, gas, cost):
	                return i
	        return -1
	
	    def isCan(self, i, gas, cost):
	        count = 0
	        N = len(gas)
	        tank = 0
	        while count < N:
	            tank = tank + gas[i] - cost[i]
	            if tank < 0:
	                return False
	            i = (i + 1) % N
	            count += 1
	        return True if tank >= 0 else False
             
                
            
            
            	      
			
### 运行结果


	Time Limit Exceeded


### 解析

从上面的例子和分析中我们可以得出几个结论：

1. 只有一种情况返回 -1 ，那就是列表 cost 的总和比列表 gas 的总和多的时候，因为这种情况一圈下来汽油的消耗大于增加，肯定跑不完一圈

2. 那为什么列表 gas 的总和大于等于 cost 的总和的时候肯定有答案呢？因为 gas 和 cost 的长度相等，那么肯定有某些索引的 gas 大于等于 cost ，随便找出这样一个索引之后，继续前进，这样就会有可能有多余的汽油节省下来，以支持我们继续向前。我看了网上有个[大佬的解释](https://www.cnblogs.com/boring09/p/4248482.html)很有意思，可以从数学的角度去解释这种现象:

		如果一个数组的总和非负，那么一定可以找到一个起始位置，从他开始绕数组一圈，累加和一直都是非负的
	
	我们假设有一个剩余油列表 remain ， remain[i] 为 gas[i]-cost[i] ，最后得到的这个列表如果满足上述的理论那么就说明肯定有解。

3. 上面的只是肯定有解的结论，我们还需要解题思路，怎么去找这个索引？我们会发现如果从 x 点开始出发，行到 y 点就走不下去的时候，那么从 x 点到 y 点之间的任意一点出发，都不可能超过 y 点。那么我们从索引 0 开始试着出发，当到达某个 i 索引油量亏损的情况下，我们就假定 [0,i] 中不可能有答案，所以我们从 i+1 作为起始点开始尝试行走，一直进行下去，直到找到那答案。只需要遍历一次就够了，所以时间复杂度为 O(n) 。

4. 这个其实是个经验，当限制条件 n 在 10^5 左右大小的时候，基本上只能使用时间复杂度为 O(n) 的算法，往这个方向靠近即可。



### 解答

	class Solution(object):
	    def canCompleteCircuit(self, gas, cost):
	        """
	        :type gas: List[int]
	        :type cost: List[int]
	        :rtype: int
	        """
	        if sum(cost) > sum(gas): return -1
	        N = len(gas)
	        idx = 0
	        tank = 0
	        result = 0
	        while idx < N:
	            tank += gas[idx] - cost[idx]
	            if tank < 0:
	                tank = 0
	                result = idx + 1
	            idx += 1
	        return result

### 运行结果

	Runtime: 540 ms, faster than 37.02% of Python online submissions for Gas Station.
	Memory Usage: 18.2 MB, less than 96.37% of Python online submissions for Gas Station.

### 原题链接

https://leetcode.com/problems/gas-station/

### 每日经典

《》 ——（）


您的支持是我最大的动力
