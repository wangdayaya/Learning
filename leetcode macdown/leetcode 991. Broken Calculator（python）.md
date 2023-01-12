leetcode  991. Broken Calculator（python）




### 描述


There is a broken calculator that has the integer startValue on its display initially. In one operation, you can:

* multiply the number on display by 2, or
* subtract 1 from the number on display.

Given two integers startValue and target, return the minimum number of operations needed to display target on the calculator.


Example 1:

	Input: startValue = 2, target = 3
	Output: 2
	Explanation: Use double operation and then decrement operation {2 -> 4 -> 3}.






Note:

1 <= x, y <= 10^9


### 解析


根据题意，有一个损坏的计算器，最初在其显示屏上显示整数 startValue。 每次我们只能进行以下任意一个操作：

* 将显示的数字乘以 2，或
* 从显示的数字中减去 1。

给定两个整数 startValue 和 target，返回在计算器上显示 target 所需的最小操作数。

一看这种题肯定就是考察贪心算法的，我们只能进行乘 2 和减 1 的操作来保证从 startValue 到 target 的最小操作次数，我们知道乘 2 操作肯定是能大幅度减小操作次数，所以我们尽量在条件允许的情况下多用乘 2 ，大体思路就是这样，但是我觉得反向操作比较好理解（个人偏好），所以我们从 target 变成 startValue ，这样我们就只能用除 2 和加 1 操作了。

我们只需要在一个循环里，不断调整 target 的大小，努力让其和 startValue 相等即可。每次循环过程中，我们判断如果 target 为奇数，那肯定要让 target 加一（这样才能保证成为偶数进行除 2 操作），结果 result 加一进行计数。此时如果 target 大于 startValue 那么就直接对 target 进行除 2 操作，结果 result 加一进行计数；否则那就说明 target 小于 startValue ，直接用结果 result 加上差值 (startValue - target) 进行计数，然后将 target 变成 startValue 即可。循环结束之后直接返回 result 即可。

时间复杂度为 O(logtarget) ，空间复杂度为 O(1)  。

### 解答
				

	class Solution(object):
	    def brokenCalc(self, startValue, target):
	        result = 0
	        while target != startValue:
	            if target % 2:
	                target += 1
	                result += 1
	            if target > startValue:
	                target //= 2
	                result += 1
	            else:
	                result += (startValue - target)
	                target = startValue
	        return result
	        
        
            	      
			
### 运行结果

	Runtime: 19 ms, faster than 87.50% of Python online submissions for Broken Calculator.
	Memory Usage: 13.4 MB, less than 75.00% of Python online submissions for Broken Calculator.

### 原题链接


https://leetcode.com/problems/broken-calculator/


您的支持是我最大的动力
