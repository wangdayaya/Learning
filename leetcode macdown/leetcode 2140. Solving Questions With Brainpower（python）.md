leetcode  2140. Solving Questions With Brainpower（python）

由亚马逊公司赞助的 Leetcode Weekly Contest 276 ，优秀者还能获得亚马逊公司的面试机会（慕了），看了一下榜单第一名是位北大的选手，为国争光，只要第一名国籍是中国的就好，吾辈楷模，我的水平令人惭愧。本文介绍的是周赛第三道题目，难度 Medium ，耗时一个小时，感觉也不是很难，考察的是动态规划，但是我用递归解题超时了，最后也没优化出来。




### 描述


You are given a 0-indexed 2D integer array questions where questions[i] = [points<sub>i</sub>, brainpower<sub>i</sub>].

The array describes the questions of an exam, where you have to process the questions in order (i.e., starting from question 0) and make a decision whether to solve or skip each question. Solving question i will earn you points<sub>i</sub> points but you will be unable to solve each of the next brainpower<sub>i</sub> questions. If you skip question i, you get to make the decision on the next question.

* For example, given questions = [[3, 2], [4, 3], [4, 4], [2, 5]]:
* If question 0 is solved, you will earn 3 points but you will be unable to solve questions 1 and 2.
* If instead, question 0 is skipped and question 1 is solved, you will earn 4 points but you will be unable to solve questions 2 and 3.

Return the maximum points you can earn for the exam.



Example 1:

	Input: questions = [[3,2],[4,3],[4,4],[2,5]]
	Output: 5
	Explanation: The maximum points can be earned by solving questions 0 and 3.
	- Solve question 0: Earn 3 points, will be unable to solve the next 2 questions
	- Unable to solve questions 1 and 2
	- Solve question 3: Earn 2 points
	Total points earned: 3 + 2 = 5. There is no other way to earn 5 or more points.

	
Example 2:

	Input: questions = [[1,1],[2,2],[3,3],[4,4],[5,5]]
	Output: 7
	Explanation: The maximum points can be earned by solving questions 1 and 4.
	- Skip question 0
	- Solve question 1: Earn 2 points, will be unable to solve the next 2 questions
	- Unable to solve questions 2 and 3
	- Solve question 4: Earn 5 points
	Total points earned: 2 + 5 = 7. There is no other way to earn 7 or more points.


Note:


	1 <= questions.length <= 105
	questions[i].length == 2
	1 <= pointsi, brainpoweri <= 105

### 解析

根据题意，有一个 0 索引的 2D 整数数组 questions ，其中 questions[i] = [points<sub>i</sub>, brainpower<sub>i</sub>]。该数组描述了考试的问题，必须按顺序从左到右处理问题，但是对于每个问题可以决定是解决还是跳过。 如果选择解决 questions[i] 将赢得 points<sub>i</sub> 个积分，但将无法解决接下来的 brainpower<sub>i</sub> 问题。 如果你跳过第 i 题，你就可以去处理下一个问题。

* 例如，给定 questions = [[3, 2], [4, 3], [4, 4], [2, 5]] ：
* 如果解决了第 0 题，将获得 3 分，但将无法解决第 1 题和第 2 题
* 如果跳过第 0 题去解决第 1 题，您将获得 4 分，但您将无法解决第 2 题和第 3 题。

最后返回通过考试获得的最高分数。

* 首先我先用 pos 去找到如果解决第 i 个问题，下一个可以解决的问题的索引，
* 然后使用 DFS 函数，函数表示从 i 个位置开始解决问题，可以得到的最大的分值
* 遍历所有问题的位置，计算 dfs(j, questions[j][0]) ，得到最后的最大的分值

我的解法是使用了递归 ，但是超时了，因为使用了 DFS 和两次循环，导致时间复杂度成了 O(n^3)  。
### 解答
				

	class Solution(object):
	    def mostPoints(self, questions):
	        """
	        :type questions: List[List[int]]
	        :rtype: int
	        """
	        pos = []
	        N = len(questions)
	        for i,(coin, brainpower) in enumerate(questions):
	            pos.append(min(i+brainpower+1, N))
	        def dfs(i, coins):
	            if i >= N - 1 or pos[i]>=N:
	                return coins
	            return max([dfs(j, coins + questions[j][0]) for j in range(pos[i], N)])
	
	        return max([dfs(j, questions[j][0]) for j in range(N)])
	        
            	      
			
### 运行结果

	Time Limit Exceeded


### 解析

看了其他大佬的解法，我发现我的思路应该变一下，不应该从前到后，而是从后向前，对于每道题目我们都知道有两种方案，一种是跳过去执行下一个题目，另一种是解决当前的问题而跳过后面若干到题目，要求我们最后得到的分数最大，这是典型的动态规划题目，我们使用动态规划的思路，从后往前遍历 questions ，定义 dp[i] 为 questions[i:] 这个子字符串能收集到的最大分数，长度为 N+1 ，因为我们要使用 dp[N] 来保存超过 questions 长度的索引的值为 0 ，动态规划的公式为 

	dp[i] = max(points + dp[min(jump + i + 1, len(questions))], dp[i + 1])

从后往前遍历结束之后，返回 dp[0] 即可。时间复杂度是 O(n) ，空间复杂度是 O(n) 。


### 解答

	class Solution(object):
	    def mostPoints(self, questions):
	        """
	        :type questions: List[List[int]]
	        :rtype: int
	        """
	        dp = [0] * (len(questions) + 1) 
	        for i in range(len(questions) - 1, -1, -1):
	            points, jump = questions[i][0], questions[i][1]
	            dp[i] = max(points + dp[min(jump + i + 1, len(questions))], dp[i + 1])
	        return dp[0]


### 运行结果

	Runtime: 2241 ms
	Memory Usage: 65.5 MB


原题链接：https://leetcode.com/contest/weekly-contest-276/problems/solving-questions-with-brainpower/



### 每日经典

《道德经》 ——老子（春秋）

谷神不死，是谓玄牝。玄牝之门，是谓天地根。绵绵若存，用之不勤。
