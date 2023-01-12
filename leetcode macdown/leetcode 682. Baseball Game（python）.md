leetcode  682. Baseball Game（python）

### 描述


You are keeping score for a baseball game with strange rules. The game consists of several rounds, where the scores of past rounds may affect future rounds' scores.

At the beginning of the game, you start with an empty record. You are given a list of strings ops, where ops[i] is the i<sup>th</sup> operation you must apply to the record and is one of the following:

* An integer x - Record a new score of x.
* "+" - Record a new score that is the sum of the previous two scores. It is guaranteed there will always be two previous scores.
* "D" - Record a new score that is double the previous score. It is guaranteed there will always be a previous score.
* "C" - Invalidate the previous score, removing it from the record. It is guaranteed there will always be a previous score.

Return the sum of all the scores on the record.


Example 1:


	Input: ops = ["5","2","C","D","+"]
	Output: 30
	Explanation:
	"5" - Add 5 to the record, record is now [5].
	"2" - Add 2 to the record, record is now [5, 2].
	"C" - Invalidate and remove the previous score, record is now [5].
	"D" - Add 2 * 5 = 10 to the record, record is now [5, 10].
	"+" - Add 5 + 10 = 15 to the record, record is now [5, 10, 15].
	The total sum is 5 + 10 + 15 = 30.
	
Example 2:

	Input: ops = ["5","-2","4","C","D","9","+","+"]
	Output: 27
	Explanation:
	"5" - Add 5 to the record, record is now [5].
	"-2" - Add -2 to the record, record is now [5, -2].
	"4" - Add 4 to the record, record is now [5, -2, 4].
	"C" - Invalidate and remove the previous score, record is now [5, -2].
	"D" - Add 2 * -2 = -4 to the record, record is now [5, -2, -4].
	"9" - Add 9 to the record, record is now [5, -2, -4, 9].
	"+" - Add -4 + 9 = 5 to the record, record is now [5, -2, -4, 9, 5].
	"+" - Add 9 + 5 = 14 to the record, record is now [5, -2, -4, 9, 5, 14].
	The total sum is 5 + -2 + -4 + 9 + 5 + 14 = 27.


Example 3:

	
	Input: ops = ["1"]
	Output: 1
	



Note:

	1 <= ops.length <= 1000
	ops[i] is "C", "D", "+", or a string representing an integer in the range [-3 * 10^4, 3 * 10^4].
	For operation "+", there will always be at least two previous scores on the record.
	For operations "C" and "D", there will always be at least one previous score on the record.


### 解析


根据题意，就是在进行一场规则特殊的棒球比赛得分记录，过去的分数会影响未来的分数，给出一个操作列表 ops ，ops[i] 有以下几种形式，规则如下：

* 整数 x - 记录 x 的新分数
* "+" ：记录一个新的分数，它是前两个分数的总和。 保证总会有两个以前的分数
* “D” ：记录一个新的分数，该分数是先前分数的两倍。 保证总会有以前的分数
* "C" ：使之前的分数无效，将其从记录中删除。 保证总会有以前的分数

返回记录上所有分数的总和。

这道题很简单，按照题意写代码即可。


### 解答
				

	class Solution(object):
	    def calPoints(self, ops):
	        """
	        :type ops: List[str]
	        :rtype: int
	        """
	        result = []
	        for i, op in enumerate(ops):
	            if op == '+':
	                result.append(result[-2] + result[-1])
	            elif op == 'D':
	                result.append(result[-1] * 2)
	            elif op == 'C':
	                result.pop(-1)
	            else:
	                result.append(int(op))
	        return sum(result)
            	      
			
### 运行结果

	Runtime: 32 ms, faster than 48.57% of Python online submissions for Baseball Game.
	Memory Usage: 13.9 MB, less than 60.95% of Python online submissions for Baseball Game.



原题链接：https://leetcode.com/problems/baseball-game/



您的支持是我最大的动力
