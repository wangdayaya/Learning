leetcode  419. Battleships in a Board（python）

### 描述

Given an m x n matrix board where each cell is a battleship 'X' or empty '.', return the number of the battleships on board.

Battleships can only be placed horizontally or vertically on board. In other words, they can only be made of the shape 1 x k (1 row, k columns) or k x 1 (k rows, 1 column), where k can be of any size. At least one horizontal or vertical cell separates between two battleships (i.e., there are no adjacent battleships).


Example 1:

![](https://assets.leetcode.com/uploads/2021/04/10/battelship-grid.jpg)

	Input: board = [["X",".",".","X"],[".",".",".","X"],[".",".",".","X"]]
	Output: 2

	
Example 2:

	Input: board = [["."]]
	Output: 0




Note:

	m == board.length
	n == board[i].length
	1 <= m, n <= 200
	board[i][j] is either '.' or 'X'.


### 解析

根据题意，其实我也没有看懂是在说什么，单词都认识，但组合起来的意思却不认识。就是有一个 m * n 的甲板上面有很多格子，每个格子可能是 X 表示有战列舰，可能是空的表示啥都没有，题目要求战列舰只能是垂直或者水平的放置，并且形状为 1 * k 或者 k * 1 （这句话我彻底懵逼了，结合例子一中的图片，我反正是没搞懂是在干啥，我觉得可能是战列舰有大有小，可能是形状为 1 * k 或者 k * 1 的大小），战列舰之间也不能相连，求在甲板上的战列舰的个数。思路比较简单：

遍历矩阵的每个位置，如果该位置为 X ，如果在位置合法的情况下，它的左边或者上边有 X ，那么不符合题目中的战舰不能相连的条件，直接进行下一个位置的判断，否则将结果 result 加一，遍历结束得到的 result 即为答案。


### 解答
				
	
	class Solution(object):
	    def countBattleships(self, board):
	        """
	        :type board: List[List[str]]
	        :rtype: int
	        """
	        result = 0
	        for i in range(len(board)):
	            for j in range(len(board[0])):
	                if board[i][j]=='X':
	                    if i>0 and board[i-1][j]=='X' or j>0 and board[i][j-1]=='X':
	                        continue
	                    result += 1
	        return result
            	      
			
### 运行结果

	
	Runtime: 52 ms, faster than 91.46% of Python online submissions for Battleships in a Board.
	Memory Usage: 16.9 MB, less than 71.36% of Python online submissions for Battleships in a Board.
	
原题链接：https://leetcode.com/problems/battleships-in-a-board/



您的支持是我最大的动力
