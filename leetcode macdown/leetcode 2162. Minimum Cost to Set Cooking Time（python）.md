leetcode  2162. Minimum Cost to Set Cooking Time（python）

「这是我参与2022首次更文挑战的第N天，活动详情查看：[2022首次更文挑战](https://juejin.cn/post/7052884569032392740 "https://juejin.cn/post/7052884569032392740")」

### 前言

这是 Biweekly Contest 71 比赛的第三题，难度 Medium  ，考察的是对实际问题的理解能力，思路对了也不是很难，我比赛的时候也是仅仅止步于此，没有时间做第四题了。



### 描述


A generic microwave supports cooking times for:

* at least 1 second.
* at most 99 minutes and 99 seconds.

To set the cooking time, you push at most four digits. The microwave normalizes what you push as four digits by prepending zeroes. It interprets the first two digits as the minutes and the last two digits as the seconds. It then adds them up as the cooking time. For example,

* You push 9 5 4 (three digits). It is normalized as 0954 and interpreted as 9 minutes and 54 seconds.
* You push 0 0 0 8 (four digits). It is interpreted as 0 minutes and 8 seconds.
* You push 8 0 9 0. It is interpreted as 80 minutes and 90 seconds.
* You push 8 1 3 0. It is interpreted as 81 minutes and 30 seconds.

You are given integers startAt, moveCost, pushCost, and targetSeconds. Initially, your finger is on the digit startAt. Moving the finger above any specific digit costs moveCost units of fatigue. Pushing the digit below the finger once costs pushCost units of fatigue.

There can be multiple ways to set the microwave to cook for targetSeconds seconds but you are interested in the way with the minimum cost.Return the minimum cost to set targetSeconds seconds of cooking time.Remember that one minute consists of 60 seconds.


Example 1:

![](https://assets.leetcode.com/uploads/2021/12/30/1.png)

	Input: startAt = 1, moveCost = 2, pushCost = 1, targetSeconds = 600
	Output: 6
	Explanation: The following are the possible ways to set the cooking time.
	- 1 0 0 0, interpreted as 10 minutes and 0 seconds.
	  The finger is already on digit 1, pushes 1 (with cost 1), moves to 0 (with cost 2), pushes 0 (with cost 1), pushes 0 (with cost 1), and pushes 0 (with cost 1).
	  The cost is: 1 + 2 + 1 + 1 + 1 = 6. This is the minimum cost.
	- 0 9 6 0, interpreted as 9 minutes and 60 seconds. That is also 600 seconds.
	  The finger moves to 0 (with cost 2), pushes 0 (with cost 1), moves to 9 (with cost 2), pushes 9 (with cost 1), moves to 6 (with cost 2), pushes 6 (with cost 1), moves to 0 (with cost 2), and pushes 0 (with cost 1).
	  The cost is: 2 + 1 + 2 + 1 + 2 + 1 + 2 + 1 = 12.
	- 9 6 0, normalized as 0960 and interpreted as 9 minutes and 60 seconds.
	  The finger moves to 9 (with cost 2), pushes 9 (with cost 1), moves to 6 (with cost 2), pushes 6 (with cost 1), moves to 0 (with cost 2), and pushes 0 (with cost 1).
	  The cost is: 2 + 1 + 2 + 1 + 2 + 1 = 9.

	
Example 2:

![](https://assets.leetcode.com/uploads/2021/12/30/2.png)

	Input: startAt = 0, moveCost = 1, pushCost = 2, targetSeconds = 76
	Output: 6
	Explanation: The optimal way is to push two digits: 7 6, interpreted as 76 seconds.
	The finger moves to 7 (with cost 1), pushes 7 (with cost 2), moves to 6 (with cost 1), and pushes 6 (with cost 2). The total cost is: 1 + 2 + 1 + 2 = 6
	Note other possible ways are 0076, 076, 0116, and 116, but none of them produces the minimum cost.

Note:


* 0 <= startAt <= 9
* 1 <= moveCost, pushCost <= 10^5
* 1 <= targetSeconds <= 6039

### 解析

根据题意，通用微波炉支持烹饪时间至少 1 秒，最多99分99秒。要设置烹饪时间，最多按四位数，微波炉可以自动在前面加上零将设定的时间标准化为四位数。前两位数表示为分钟，后两位数表示为秒。然后将它们加起来作为烹饪时间。

另外还给出了整数 startAt、moveCost、pushCost 和 targetSeconds。最初的时候手指在数字 startAt 上。将手指移动到任何特定数字上方都会消耗 moveCost 个能量，将手指按下按钮一次会消耗 pushCost 个能量。我们有多种方法可以将微波炉设置为 targetSeconds 秒，但我们要返回最少的能量单位。

这道题很贴近实际的应用，所以比较有意思，感觉不到在考什么，可能是在考察对题目的理解能力吧。

我们先考虑假定有时间目标的情况下，如何求出消耗的能量数量，这还是比较简单的，定义一个函数 cost ，参数为 s 表示时间的字符串，初始化 result 为 0 ，表示总共消耗的能量，当我们的手一开始的位置等于  startAt ，就不用消耗移动手的能量，否则要计算移动手的能力，然后再加上按按钮的能力即可。然后从第二个数字开始，如果当前的数字和前一个数字相等，那么我们又节省了移动手的能力，直接计算按按钮的能力，否则我们就要加上移动手的能量和按按钮的能量，最后返回 result 即可。

其实上面计算能量的函数比较简单，比较麻烦的是这个微波炉对 targetSeconds 有不同的表现形式。分钟数肯定最多只有 M=targetSeconds // 60 ，遍历 range(M , -1, -1) ，如果 targetSeconds - i * 60 < 100 ，说明这种分钟加秒数是可以成立的，我们将其标准化为可用的模式之后加入列表 nums 中，否则直接跳出循环，然后遍历 nums 中的各种形式，找出消耗能量最小的值即可。




### 解答
				
	
	class Solution(object):
	    def minCostSetTime(self, startAt, moveCost, pushCost, targetSeconds):
	        """
	        :type startAt: int
	        :type moveCost: int
	        :type pushCost: int
	        :type targetSeconds: int
	        :rtype: int
	        """
	        def cost(s):
	            result = 0
	            if startAt == int(s[0]):
	                result += 0
	            else:
	                result += moveCost
	            result += pushCost
	            for i in range(1, len(s)):
	                if s[i - 1] == s[i]:
	                    result += pushCost
	                else:
	                    result += moveCost
	                    result += pushCost
	            return result
	
	        M = targetSeconds // 60
	        nums = []
	        for i in range(M , -1, -1):
	            if targetSeconds - i * 60 < 100:
	                a = str(i)
	                b = str(targetSeconds - i * 60)
	                if b == '0':
	                    r = a + '00'
	                elif len(b) == 1:
	                    r = a + '0' + b
	                else:
	                    r = a + b
	                if len(r)<=4:
	                    nums.append(r.lstrip('0'))
	            else:
	                break
	        result = float('inf')
	
	        for r in nums:
	            result = min(result, cost(r))
	        return result
			
### 运行结果

	225 / 225 test cases passed.
	Status: Accepted
	Runtime: 26 ms
	Memory Usage: 13.3 MB


### 原题链接


https://leetcode.com/contest/biweekly-contest-71/problems/minimum-cost-to-set-cooking-time/

您的支持是我最大的动力
