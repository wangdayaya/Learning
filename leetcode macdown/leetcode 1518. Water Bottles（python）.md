leetcode  1518. Water Bottles（python）

### 描述


Given numBottles full water bottles, you can exchange numExchange empty water bottles for one full water bottle.

The operation of drinking a full water bottle turns it into an empty bottle.

Return the maximum number of water bottles you can drink.


Example 1:

![avatar](https://assets.leetcode.com/uploads/2020/07/01/sample_1_1875.png)

	Input: numBottles = 9, numExchange = 3
	Output: 13
	Explanation: You can exchange 3 empty bottles to get 1 full water bottle.
	Number of water bottles you can drink: 9 + 3 + 1 = 13.

	
Example 2:

![avatar](https://assets.leetcode.com/uploads/2020/07/01/sample_2_1875.png)

	Input: numBottles = 15, numExchange = 4
	Output: 19
	Explanation: You can exchange 4 empty bottles to get 1 full water bottle. 
	Number of water bottles you can drink: 15 + 3 + 1 = 19.

Example 3:


	Input: numBottles = 5, numExchange = 5
	Output: 6
	
Example 4:

	Input: numBottles = 2, numExchange = 3
	Output: 2

	


Note:

	
	1 <= numBottles <= 100
	2 <= numExchange <= 100


### 解析

根据题意，就是给出 numBottles 个满水瓶子，喝完之后每 numExchange 个空瓶子能换一个满水瓶子继续喝，问不断喝光换瓶子一直到不能换为止，一共喝了多少瓶水。其实比较简单，定义变量 r 表示一共喝了多少水，开始肯定喝了 numBottles 瓶水，所以 numBottles 赋值给 r ，循环当 numBottles 大于等于 numExchange 的时候，不断用 numExchange 对 numBottles 整除和取模操作，整除表示能用空瓶子换的 t
 瓶满水瓶，结果 r 又多了 t 瓶水，取模表示不够换满水瓶的剩余 m 瓶空瓶子，然后将 m+t 赋值给 numBottles 表示还有多少空瓶子，继续判断循环条件，循环结束就可以得到结果。



### 解答
				

	class Solution(object):
	    def numWaterBottles(self, numBottles, numExchange):
	        """
	        :type numBottles: int
	        :type numExchange: int
	        :rtype: int
	        """
	        r = numBottles
	        while numBottles>=numExchange:
	            t = numBottles//numExchange
	            m = numBottles%numExchange
	            numBottles = m + t
	            r += t
	        return r			
	        
### 运行结果

	Runtime: 16 ms, faster than 72.96% of Python online submissions for Water Bottles.
	Memory Usage: 13 MB, less than 99.57% of Python online submissions for Water Bottles.


原题链接：https://leetcode.com/problems/water-bottles/



您的支持是我最大的动力
