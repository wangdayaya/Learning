leetcode  1275. Find Winner on a Tic Tac Toe Game（python）

### 描述

Tic-tac-toe is played by two players A and B on a 3 x 3 grid. The rules of Tic-Tac-Toe are:

* Players take turns placing characters into empty squares ' '.
* The first player A always places 'X' characters, while the second player B always places 'O' characters.
* 'X' and 'O' characters are always placed into empty squares, never on filled ones.
* The game ends when there are three of the same (non-empty) character filling any row, column, or diagonal.
* The game also ends if all squares are non-empty.
* No more moves can be played if the game is over.

Given a 2D integer array moves where moves[i] = [row<sub>i</sub>, col<sub>i</sub>] indicates that the i<sup>th</sup> move will be played on grid[row<sub>i</sub>][col<sub>i</sub>]. return the winner of the game if it exists (A or B). In case the game ends in a draw return "Draw". If there are still movements to play return "Pending".

You can assume that moves is valid (i.e., it follows the rules of Tic-Tac-Toe), the grid is initially empty, and A will play first.

 



Example 1:

![](https://assets.leetcode.com/uploads/2021/09/22/xo1-grid.jpg)

	Input: moves = [[0,0],[2,0],[1,1],[2,1],[2,2]]
	Output: "A"
	Explanation: A wins, they always play first.

	
Example 2:


![](https://assets.leetcode.com/uploads/2021/09/22/xo2-grid.jpg)

	Input: moves = [[0,0],[1,1],[0,1],[0,2],[1,0],[2,0]]
	Output: "B"
	Explanation: B wins.

Example 3:

![](https://assets.leetcode.com/uploads/2021/09/22/xo3-grid.jpg)
	
	Input: moves = [[0,0],[1,1],[2,0],[1,0],[1,2],[2,1],[0,1],[0,2],[2,2]]
	Output: "Draw"
	Explanation: The game ends in a draw since there are no moves to make.

	
Example 4:

![](https://assets.leetcode.com/uploads/2021/09/22/xo4-grid.jpg)

	Input: moves = [[0,0],[1,1]]
	Output: "Pending"
	Explanation: The game has not finished yet.




Note:

	1 <= moves.length <= 9
	moves[i].length == 2
	0 <= rowi, coli <= 2
	There are no repeated elements on moves.
	moves follow the rules of tic tac toe.


### 解析


根据题意，两个玩家在一个 3*3 的方格内玩名叫 Tic-tac-toe 的游戏，规则如下：

* 玩家轮流将角色放入空方块中。
* 第一个玩家 A 总是放置 'X' 字符，而第二个玩家 B 总是放置 'O' 字符。
* 'X' 和 'O' 字符总是被放置在空方块中
* 当三个相同（非空）字符填充任何行、列或对角线时，游戏结束。
* 如果所有方格都不为空，游戏也会结束。
* 如果游戏结束，就不能再进行任何动作。

给定一个二维整数数组 moves ，其中 move[i] = [row<sub>i</sub>,col<sub>i</sub>] 表示第 i 个操作在 grid[row<sub>i</sub>][col<sub>i</sub>] 上进行。 如果 A 或 B 能满足三个连续字符在同行、同列或者对角线相同，则返回游戏的获胜者。 如果游戏以平局结束，则返回 Draw 。 如果仍有空间可要继续进行游戏，请返回 Pending 。

题目其实理解之后也很简单，思路如下：

* 将 A 在方格内的所有位置都拿出来放入列表 A 中，将 B 在方格内的所有位置都拿出来放入列表 B 中
* 因为赢的条件只有八种情况，所以都列出来放入 winCom 中
* 遍历 winCom 中的每种赢得情况 com ，如果判断列表 A 和 com 中的元素相等则直接返回 A ，同理如果判断列表 B 和 com 中的元素相等则直接返回 B
* 否则如果列表 A 和列表 B 的长度和为 9 说明平局了，直接返回 Draw
* 否则直接返回 Pending


### 解答
				
	class Solution(object):
	    def tictactoe(self, moves):
	        """
	        :type moves: List[List[int]]
	        :rtype: str
	        """
	        winCom = [[[0,0],[0,1],[0,2]] , [[1,0],[1,1],[1,2]], [[2,0],[2,1],[2,2]],
	                  [[0,0],[1,0],[2,0]],[[0,1],[1,1],[2,1]],[[0,2],[1,2],[2,2]],
	                  [[0,0],[1,1],[2,2]],[[0,2],[1,1],[2,0]]]
	        A = [c for c in moves[0:len(moves):2]]
	        B = [c for c in moves[1:len(moves):2]]
	        for com in winCom:
	            if self.isWin(A, com): return 'A'
	            if self.isWin(B, com): return 'B'
	        if len(A)+len(B)==9: return "Draw"
	        return "Pending"
	            
	    def isWin(self, L, com):
	        result = 0
	        for c in L:
	            if c in com:
	                result += 1
	        return result == 3
	            
	            
	


            	      
			
### 运行结果
	
	Runtime: 24 ms, faster than 52.05% of Python online submissions for Find Winner on a Tic Tac Toe Game.
	Memory Usage: 13.3 MB, less than 71.11% of Python online submissions for Find Winner on a Tic Tac Toe Game.


原题链接：https://leetcode.com/problems/find-winner-on-a-tic-tac-toe-game/



您的支持是我最大的动力
