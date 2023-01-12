leetcode  901. Online Stock Span（python）




### 描述

Design an algorithm that collects daily price quotes for some stock and returns the span of that stock's price for the current day. The span of the stock's price today is defined as the maximum number of consecutive days (starting from today and going backward) for which the stock price was less than or equal to today's price. Implement the StockSpanner class:

* StockSpanner() Initializes the object of the class.
* int next(int price) Returns the span of the stock's price given that today's price is price.



Example 1:

	Input
	["StockSpanner", "next", "next", "next", "next", "next", "next", "next"]
	[[], [100], [80], [60], [70], [60], [75], [85]]
	Output
	[null, 1, 1, 1, 2, 1, 4, 6]
	
	Explanation
	StockSpanner stockSpanner = new StockSpanner();
	stockSpanner.next(100); // return 1
	stockSpanner.next(80);  // return 1
	stockSpanner.next(60);  // return 1
	stockSpanner.next(70);  // return 2
	stockSpanner.next(60);  // return 1
	stockSpanner.next(75);  // return 4, because the last 4 prices (including today's price of 75) were less than or equal to today's price.
	stockSpanner.next(85);  // return 6




Note:

	1 <= price <= 10^5
	At most 104 calls will be made to next.

### 解析

根据题意，设计一种算法，该算法收集某些股票的每日报价，并返回该股票当天的价格范围。股票当天价格的跨度被定义为股票价格小于或等于当天价格的最大连续天数（从当天往前的最大连续天数，包括当天）。例如，如果未来 7 天股票的价格是 [100,80,60,70,60,75,85]，那么股票跨度将是 [1,1,1,2,1,4,6] 。

实现 StockSpanner 类：

* StockSpanner（） 初始化类的对象
* int next（int price） 给定今天的价格是 price ，返回股票价格的跨度。

本题很明显考察的是单调栈的实际应用，我们在给定当天价格的同时，计算在之前小于或者等于当天价格的连续天数，我们只需要初始化维护一个单调递减的单调栈，在 next 中对于当天跨度的计算函数逻辑如下：

* 如果当天的价格大于等于栈顶的价格，则将栈顶的（价格，跨度）元素弹出，并用计数器 count 进行计数，不断重复这个过程；
* 否则将当天的价格和跨度压入栈顶，以供后面日子的使用；
* 最后返回当天的跨度；

时间复杂度为 O(N) ，空间复杂度为 O(N) ，N 为数组大小。
### 解答

	class StockSpanner(object):
	    def __init__(self):
	        self.stack = []
	
	    def next(self, price):
	        count = 1
	        while self.stack and self.stack[-1][0] <= price:
	            count += self.stack.pop()[1]
	        self.stack.append([price, count])
	        return count

### 运行结果

	Runtime Beats 83.3%
	Memory Beats 94.55%


### 原题链接

https://leetcode.com/problems/online-stock-span/description/


您的支持是我最大的动力
