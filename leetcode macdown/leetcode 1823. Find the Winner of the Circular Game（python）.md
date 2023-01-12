leetcode  1823. Find the Winner of the Circular Game（python）

### 描述

There are n friends that are playing a game. The friends are sitting in a circle and are numbered from 1 to n in clockwise order. More formally, moving clockwise from the i<sup>th</sup> friend brings you to the (i+1)<sup>th</sup> friend for 1 <= i < n, and moving clockwise from the n<sup>th</sup> friend brings you to the 1<sup>st</sup> friend.

The rules of the game are as follows:

* Start at the 1<sup>st</sup> friend.
* Count the next k friends in the clockwise direction including the friend you started at. The counting wraps around the circle and may count some friends more than once.
* The last friend you counted leaves the circle and loses the game.
* If there is still more than one friend in the circle, go back to step 2 starting from the friend immediately clockwise of the friend who just lost and repeat.
* Else, the last friend in the circle wins the game.

Given the number of friends, n, and an integer k, return the winner of the game.

 



Example 1:

![](https://assets.leetcode.com/uploads/2021/03/25/ic234-q2-ex11.png)


	Input: n = 5, k = 2
	Output: 3
	Explanation: Here are the steps of the game:
	1) Start at friend 1.
	2) Count 2 friends clockwise, which are friends 1 and 2.
	3) Friend 2 leaves the circle. Next start is friend 3.
	4) Count 2 friends clockwise, which are friends 3 and 4.
	5) Friend 4 leaves the circle. Next start is friend 5.
	6) Count 2 friends clockwise, which are friends 5 and 1.
	7) Friend 1 leaves the circle. Next start is friend 3.
	8) Count 2 friends clockwise, which are friends 3 and 5.
	9) Friend 5 leaves the circle. Only friend 3 is left, so they are the winner.
	
Example 2:

	Input: n = 6, k = 5
	Output: 1
	Explanation: The friends leave in this order: 5, 4, 6, 2, 3. The winner is friend 1.





Note:

	1 <= k <= n <= 500


### 解析


根据题意，有 n 个人在玩一个游戏，这 n 个人坐成了一个圆圈，并且按照顺时针的顺序给每个人按照 1-n 的顺序进行标记。更正规的是按照顺时针顺序把第 i 个人带到第 i+1 的位置，1<=i<n ，并且可以从第 n 个人顺时针移动到第一个人的位置上。

游戏规则如下：

* 开始是在第一个人的位置
* 数 k 个包含自己在内的顺时针方向移动的步数，计数是绕圆进行的，可能有的人被数了多次
* 你数到的最后一个人要离开圆圈，并且也代表他输掉了游戏
* 如果仍然存在超过一个人在圆中，从刚出局输掉比赛的人的下一个顺时针的人开始，回到步骤 2 开始继续执行相关操作
* 否则，圈子中剩下的最后一个人赢得游戏


这个题的说明是有点长，不过多读几次理解了意思就觉得这道题不难了，就是一个找规律的题。思路比较简单：

* 初始化一个包含 1-n 的列表 friends ，以及列表长度 N ，其实位置的索引 index 为 0
* 当 N 大于 1 的时候执行 while 循环，通过 (index + k - 1) % N 找到要淘汰的人的索引，然后将其从列表 friends 移除，并更新列表长度 N
* 最后剩下的一个 friends 中的人就是胜利者


### 解答
				
	class Solution(object):
	    def findTheWinner(self, n, k):
	        """
	        :type n: int
	        :type k: int
	        :rtype: int
	        """
	        friends = [i for i in range(1,n+1)]
	        N = len(friends)
	        index = 0
	        while N>1:
	            index = (index + k - 1) % N
	            friends.pop(index)
	            N = len(friends)
	        return friends[0]

            	      
			
### 运行结果

	Runtime: 41 ms, faster than 45.39% of Python online submissions for Find the Winner of the Circular Game.
	Memory Usage: 13.4 MB, less than 63.12% of Python online submissions for Find the Winner of the Circular Game.


原题链接：https://leetcode.com/problems/find-the-winner-of-the-circular-game/



您的支持是我最大的动力
