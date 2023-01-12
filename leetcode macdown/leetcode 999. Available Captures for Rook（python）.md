leetcode  999. Available Captures for Rook（python）

### 描述

On an 8 x 8 chessboard, there is exactly one white rook 'R' and some number of white bishops 'B', black pawns 'p', and empty squares '.'.

When the rook moves, it chooses one of four cardinal directions (north, east, south, or west), then moves in that direction until it chooses to stop, reaches the edge of the board, captures a black pawn, or is blocked by a white bishop. A rook is considered attacking a pawn if the rook can capture the pawn on the rook's turn. The number of available captures for the white rook is the number of pawns that the rook is attacking.

Return the number of available captures for the white rook.



Example 1:


![](https://assets.leetcode.com/uploads/2019/02/20/1253_example_1_improved.PNG)

	Input: board = [[".",".",".",".",".",".",".","."],[".",".",".","p",".",".",".","."],[".",".",".","R",".",".",".","p"],[".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".","."],[".",".",".","p",".",".",".","."],[".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".","."]]
	Output: 3
	Explanation: In this example, the rook is attacking all the pawns.
	
Example 2:


![](https://assets.leetcode.com/uploads/2019/02/19/1253_example_2_improved.PNG)

	Input: board = [[".",".",".",".",".",".",".","."],[".","p","p","p","p","p",".","."],[".","p","p","B","p","p",".","."],[".","p","B","R","B","p",".","."],[".","p","p","B","p","p",".","."],[".","p","p","p","p","p",".","."],[".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".","."]]
	Output: 0
	Explanation: The bishops are blocking the rook from attacking any of the pawns.

Example 3:

![](https://assets.leetcode.com/uploads/2019/02/20/1253_example_3_improved.PNG)

	Input: board = [[".",".",".",".",".",".",".","."],[".",".",".","p",".",".",".","."],[".",".",".","p",".",".",".","."],["p","p",".","R",".","p","B","."],[".",".",".",".",".",".",".","."],[".",".",".","B",".",".",".","."],[".",".",".","p",".",".",".","."],[".",".",".",".",".",".",".","."]]
	Output: 3
	Explanation: The rook is attacking the pawns at positions b5, d6, and f5.




Note:

	board.length == 8
	board[i].length == 8
	board[i][j] is either 'R', '.', 'B', or 'p'
	There is exactly one cell with board[i][j] == 'R'



### 解析

根据题意，就是在一个 8 x 8 的棋盘上，正好有一个白车 R 和一些白棋 B 、黑棋 p 和空方 . 。

当车移动时，它选择四个基本方向（北、东、南或西）之一，然后向该方向移动，直到它选择停止、或者到达棋盘边缘、或者捕获一个黑色棋子或被一位白旗阻塞，然后去其他方向进行相同的操作，直到四个方向都执行完， 返回白车可捕获的黑棋数量。

题比较绕，因为说的不是很清楚，我也是看网上的大神解释才理解了题意，选择某一个方向走下去，终止条件就是：碰到 B 或者 p ，或者到达了边界，大体思路如下：

* 先遍历整个棋盘，没有找到 R 直接返回 0 ，如果找到了 R ，然后向四个方向做下面相同的操作
* 当遇到边界或者 B ，则停止该方向的运动，去另外的方向运动
* 当遇到 p ，则使用计数器加一，表示捕获到黑棋，并且停止该方向的运动，去另外的方向运动
* 当四个方向都走完饭回计数器即可



### 解答
				
	class Solution(object):
	    def numRookCaptures(self, board):
	        """
	        :type board: List[List[str]]
	        :rtype: int
	        """
	        def check_direction(R,C,direction):
	            if direction=="right":
	                for col in range(C+1,8): 
	                    if board[R][col]=='p':return 1
	                    if board[R][col]=='B':return 0
	            if direction=="left":
	                for col in range(C-1,-1,-1): 
	                    if board[R][col]=='p':return 1
	                    if board[R][col]=='B':return 0
	            if direction=="down":
	                for row in range(R+1,8): 
	                    if board[row][C]=='p':return 1
	                    if board[row][C]=='B':return 0
	            if direction=="up": 
	                for row in range(R-1,-1,-1): 
	                    if board[row][C]=='p':return 1
	                    if board[row][C]=='B':return 0
	            return 0
	                    
	        for row in range(8):
	            for col in range(8):
	                if board[row][col]=="R":
	                    return sum([check_direction(row,col,"right"),
	                                check_direction(row,col,"left"),
	                                check_direction(row,col,"up"),
	                                check_direction(row,col,"down")])

            	      
			
### 运行结果

	Runtime: 12 ms, faster than 96.00% of Python online submissions for Available Captures for Rook.
	Memory Usage: 13.6 MB, less than 44.00% of Python online submissions for Available Captures for Rook.




原题链接：https://leetcode.com/problems/available-captures-for-rook/



您的支持是我最大的动力
