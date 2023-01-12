leetcode  1716. Calculate Money in Leetcode Bank（python）

### 描述


Hercy wants to save money for his first car. He puts money in the Leetcode bank every day.

He starts by putting in $1 on Monday, the first day. Every day from Tuesday to Sunday, he will put in $1 more than the day before. On every subsequent Monday, he will put in $1 more than the previous Monday.

Given n, return the total amount of money he will have in the Leetcode bank at the end of the nth day.


Example 1:

	Input: n = 4
	Output: 10
	Explanation: After the 4th day, the total is 1 + 2 + 3 + 4 = 10.

	
Example 2:


	Input: n = 10
	Output: 37
	Explanation: After the 10th day, the total is (1 + 2 + 3 + 4 + 5 + 6 + 7) + (2 + 3 + 4) = 37. Notice that on the 2nd Monday, Hercy only puts in $2.

Example 3:


	Input: n = 20
	Output: 96
	Explanation: After the 20th day, the total is (1 + 2 + 3 + 4 + 5 + 6 + 7) + (2 + 3 + 4 + 5 + 6 + 7 + 8) + (3 + 4 + 5 + 6 + 7 + 8) = 96.
	



Note:

	1 <= n <= 1000


### 解析

根据题意，就是第一周按照【1，2，3，4，5，6，7】的方式存钱，第二周按照【2，3，4，5，6，7，8】的方式存钱，第三周按照【3，4，5，6，7，8，9】，第四周按照【4，5，6，7，8，9，10】等等规律。以此类推，这就是一个找规律的题目，找出规律就会写代码了。


### 解答
				

	class Solution(object):
	    def totalMoney(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        result = 0
	        base = 1
	        for i in range(1, n+1):
	            if i%7:
	                result += (base-1)+i%7
	            else:
	                result += base+(i-1)%7
	                base += 1
	        return result
            	      
			
### 运行结果

	Runtime: 28 ms, faster than 23.66% of Python online submissions for Calculate Money in Leetcode Bank.
	Memory Usage: 13.4 MB, less than 28.12% of Python online submissions for Calculate Money in Leetcode Bank.

### 思考

这个题给了我一个思考，加入我是这个人 Hercy ，我想自己买车，也按照他这种方式存钱，我得需要多少周的时间才能完成，我的理想是买一辆野马，需要 40 万左右，下面开始写代码。

### 解答

	def cal(n):
	    result = 0
	    base = [1,2,3,4,5,6,7]
	    for i in range(n//sum(base)):
	        t = sum(base) + i * 7
	        result += t
	        print('这是第%d周，这周攒了%d元，累计攒了%d元'%(i+1,t,result))
	        if result >= n:
	            return i+1
	            
### 运行结果

	...
	这是第331周，这周攒了2338元，累计攒了391573元
	这是第332周，这周攒了2345元，累计攒了393918元
	这是第333周，这周攒了2352元，累计攒了396270元
	这是第334周，这周攒了2359元，累计攒了398629元
	这是第335周，这周攒了2366元，累计攒了400995元

这里只打印了部分结果，从结果上来看只需要 335 周的时间，也就是六年半的时间，这么一看好像我离买车的目标还是很好实现的，毕竟实际情况这六年我的收入不断增长的。

原题链接：https://leetcode.com/problems/calculate-money-in-leetcode-bank/



您的支持是我最大的动力
