leetcode  2280. Minimum Lines to Represent a Line Chart（python）



### 描述



You are given a 2D integer array stockPrices where stockPrices[i] = [dayi, pricei] indicates the price of the stock on day dayi is pricei. A line chart is created from the array by plotting the points on an XY plane with the X-axis representing the day and the Y-axis representing the price and connecting adjacent points. One such example is shown below:

![](https://assets.leetcode.com/uploads/2022/03/30/1920px-pushkin_population_historysvg.png)


Example 1:

![](https://assets.leetcode.com/uploads/2022/03/30/ex0.png)

	Input: stockPrices = [[1,7],[2,6],[3,5],[4,4],[5,4],[6,3],[7,2],[8,1]]
	Output: 3
	Explanation:
	The diagram above represents the input, with the X-axis representing the day and Y-axis representing the price.
	The following 3 lines can be drawn to represent the line chart:
	- Line 1 (in red) from (1,7) to (4,4) passing through (1,7), (2,6), (3,5), and (4,4).
	- Line 2 (in blue) from (4,4) to (5,4).
	- Line 3 (in green) from (5,4) to (8,1) passing through (5,4), (6,3), (7,2), and (8,1).
	It can be shown that it is not possible to represent the line chart using less than 3 lines.

	
Example 2:

![](https://assets.leetcode.com/uploads/2022/03/30/ex1.png)

	Input: stockPrices = [[3,4],[1,2],[7,8],[2,3]]
	Output: 1
	Explanation:
	As shown in the diagram above, the line chart can be represented with a single line.


 


Note:

	1 <= stockPrices.length <= 10^5
	stockPrices[i].length == 2
	1 <= dayi, pricei <= 10^9
	All dayi are distinct.



### 解析

根据题意，给定一个二维整数数组 stockPrices ，其中 stockPrices[i] = [dayi, pricei] 表示第 i 天的股票价格为 pricei 。 通过在 XY 平面上绘制点来按照数组数据创建折线图，其中 X 轴代表当天，Y 轴代表价格并连接相邻点。 返回将所有点都连接起来所使用的直线最小数量。

我们分析题目就能发现，能减少直线数量的关键就在于，相邻的三个点中前两个点和后两个点能不能是斜率一样的，如果能那么就不用增加直接的数量，否则斜率不一样的情况下肯定要增加不用的直线数量。

比赛的时候我有点疯了，直接按照斜率公式去判断两个斜率是否相等，并给出了斜率相等的容忍误差，但是这仍然无法满足题意，因为题目中的数都太大了，一般的容忍误差无法做到。其实最简单的方法就是将原始的公式

	dy / dx == dyy / dxx
	
变化为，这样就能避免精度损失：

	dy * dxx == dyy * dx
	
时间复杂度为 O(N) ，空间复杂度为 O(1) 。



### 解答
				

	class Solution(object):
	    def minimumLines(self, stockPrices):
	        """
	        :type stockPrices: List[List[int]]
	        :rtype: int
	        """
	        N = len(stockPrices)
	        if N == 1: return 0
	        if N == 2: return 1
	        result = 1
	        stockPrices.sort()
	        for i in range(1, N):
	            if -1<i-1 and i+1<N:
	                dx = stockPrices[i][0] - stockPrices[i-1][0]
	                dy = stockPrices[i][1] - stockPrices[i-1][1]
	                dxx = stockPrices[i+1][0] - stockPrices[i][0]
	                dyy = stockPrices[i+1][1] - stockPrices[i][1]
	                if dy*dxx == dyy * dx:
	                    continue
	                else:
	                    result += 1
	        return result
            	      
			
### 运行结果


	79 / 79 test cases passed.
	Status: Accepted
	Runtime: 2340 ms
	Memory Usage: 61.6 MB


### 原题链接

https://leetcode.com/contest/weekly-contest-294/problems/minimum-lines-to-represent-a-line-chart/



您的支持是我最大的动力
