leetcode  1626. Best Team With No Conflicts（python）




### 描述


You are the manager of a basketball team. For the upcoming tournament, you want to choose the team with the highest overall score. The score of the team is the sum of scores of all the players in the team. However, the basketball team is not allowed to have conflicts. A conflict exists if a younger player has a strictly higher score than an older player. A conflict does not occur between players of the same age. Given two lists, scores and ages, where each scores[i] and ages[i] represents the score and age of the i<sup>th</sup> player, respectively, return the highest overall score of all possible basketball teams.


Example 1:

	Input: scores = [1,3,5,10,15], ages = [1,2,3,4,5]
	Output: 34
	Explanation: You can choose all the players.

	
Example 2:


	Input: scores = [4,5,6,5], ages = [2,1,2,1]
	Output: 16
	Explanation: It is best to choose the last 3 players. Notice that you are allowed to choose multiple people of the same age.

Example 3:


	Input: scores = [1,2,3,5], ages = [8,9,10,1]
	Output: 6
	Explanation: It is best to choose the first 3 players. 


Note:


* 	1 <= scores.length, ages.length <= 1000
* 	scores.length == ages.length
* 	1 <= scores[i] <= 10^6
* 	1 <= ages[i] <= 1000

### 解析

根据题意，你是一个篮球队的经理。对于即将到来的锦标赛，希望选择总分最高的球队。球队的分数是团队中所有球员的分数之和。但是，篮球队不允许发生冲突。如果年轻球员的得分严格高于年长球员，则存在冲突。同龄玩家之间不会发生冲突。

给定两个列表 scores 和  ages ，其中每个 scores[i] 和 ages[i] 分别代表第 i 个球员的分数和年龄，返回篮球队可能无矛盾的最高总分。

因为总体的计算基础是不发生冲突，而冲突的来源是年龄，所以我们先把 scores 和  ages 按照 ages 的升序来进行排序，这样我们就能先解决年龄的排序问题，接下来只需要考虑 scores 即可。如例子 2 中，scores = [4,5,6,5] ，ages = [2,1,2,1]，排序后为：ages=[1,1,2,2]，scores=[5,5,4,6] 。剩下的工作就是挑选分数不会降低的子序列，也就是最长上升子序列 LIS ，保证球队的最大分数即可。

时间复杂度为 O(NlogN + N \* M) ，N 为 scores 的长度，M 为 ages 的长度。空间复杂度为 O(M) 。

### 解答

	class Solution:
	    def bestTeamScore(self, scores: List[int], ages: List[int]) -> int:
	        n = len(scores)
	        L = list(zip(ages, scores))
	        L.sort(key=lambda x: (x[0], x[1]))
	        dp = [L[i][1] for i in range(n)]
	        for i in range(n):
	            for j in range(i):
	                if L[i][1] >= L[j][1]:
	                    dp[i] = max(dp[i], dp[j] + L[i][1])
	        return max(dp)

### 运行结果



	Runtime 2233 ms，Beats 49.65%
	Memory 14.2 MB，Beats 83.69%

### 原题链接

	https://leetcode.com/problems/best-team-with-no-conflicts/submissions/888456244/


您的支持是我最大的动力
