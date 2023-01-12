leetcode  1222. Queens That Can Attack the King（python）

### 描述

On an 8x8 chessboard, there can be multiple Black Queens and one White King.

Given an array of integer coordinates queens that represents the positions of the Black Queens, and a pair of coordinates king that represent the position of the White King, return the coordinates of all the queens (in any order) that can attack the King.





Example 1:


![](https://assets.leetcode.com/uploads/2019/10/01/untitled-diagram.jpg)

	Input: queens = [[0,1],[1,0],[4,0],[0,4],[3,3],[2,4]], king = [0,0]
	Output: [[0,1],[1,0],[3,3]]
	Explanation:  
	The queen at [0,1] can attack the king cause they're in the same row. 
	The queen at [1,0] can attack the king cause they're in the same column. 
	The queen at [3,3] can attack the king cause they're in the same diagnal. 
	The queen at [0,4] can't attack the king cause it's blocked by the queen at [0,1]. 
	The queen at [4,0] can't attack the king cause it's blocked by the queen at [1,0]. 
	The queen at [2,4] can't attack the king cause it's not in the same row/column/diagnal as the king.
	
Example 2:

![](https://assets.leetcode.com/uploads/2019/10/01/untitled-diagram-1.jpg)

	Input: queens = [[0,0],[1,1],[2,2],[3,4],[3,5],[4,4],[4,5]], king = [3,3]
	Output: [[2,2],[3,4],[4,4]]


Example 3:

![](https://assets.leetcode.com/uploads/2019/10/01/untitled-diagram-2.jpg)

	Input: queens = [[5,6],[7,7],[2,1],[0,7],[1,6],[5,1],[3,7],[0,3],[4,0],[1,2],[6,3],[5,0],[0,4],[2,2],[1,1],[6,4],[5,4],[0,0],[2,6],[4,5],[5,2],[1,4],[7,5],[2,3],[0,5],[4,2],[1,0],[2,7],[0,1],[4,6],[6,1],[0,6],[4,3],[1,7]], king = [3,4]
	Output: [[2,3],[1,4],[1,6],[3,7],[4,3],[5,4],[4,5]]






Note:


	1 <= queens.length <= 63
	queens[i].length == 2
	0 <= queens[i][j] < 8
	king.length == 2
	0 <= king[0], king[1] < 8
	At most one piece is allowed in a cell.

### 解析


根据题意，给出一个 8x8 的棋盘，里面有多个黑皇后和一个白国王。给定一个代表黑皇后位置的整数坐标数组 queens 和一个代表白国王位置的坐标 king ，返回所有可以攻击国王的皇后的坐标（按任意顺序）。

虽然我没玩过这个游戏，不知道皇后怎么进攻国王，但是看了例子一基本就明白了，当皇后和国王在同一列或者同一行或者同一对角线上并且中间没有其他的棋子遮挡，那么皇后就可以进攻国王。知道这个就很好解题了，只需要从上下左右还有四个对角线一共八个方向，找第一个出现的皇后的位置记录到结果中并返回即可。

### 解答
				

	class Solution(object):
	    def queensAttacktheKing(self, queens, king):
	        """
	        :type queens: List[List[int]]
	        :type king: List[int]
	        :rtype: List[List[int]]
	        """
	        result = []
	        x,y = king
	        # up
	        for r in range(x-1,-1,-1):
	            if [r,y] in queens:
	                result.append([r,y])
	                break
	        # down
	        for r in range(x+1, 8):
	            if [r,y] in queens:
	                result.append([r,y])
	                break
	        
	        # left
	        for c in range(y-1,-1,-1):
	            if [x,c] in queens:
	                result.append([x,c])
	                break
	                
	        # right
	        for c in range(y+1, 8):
	            if [x,c] in queens:
	                result.append([x,c])
	                break
	                
	        tmp_x = x
	        tmp_y = y
	        # diagnal
	        for _ in range(8):
	            if tmp_x-1>=0 and tmp_y-1>=0 and [tmp_x-1,tmp_y-1] in queens:
	                result.append([tmp_x-1,tmp_y-1])
	                break
	            else:
	                tmp_x -= 1
	                tmp_y -= 1
	                
	                
	        tmp_x = x
	        tmp_y = y
	        # diagnal
	        for _ in range(8):
	            if tmp_x+1<8 and tmp_y+1<8 and [tmp_x+1,tmp_y+1] in queens:
	                result.append([tmp_x+1,tmp_y+1])
	                break
	            else:
	                tmp_x += 1
	                tmp_y += 1
	                
	        tmp_x = x
	        tmp_y = y
	        # diagnal
	        for _ in range(8):
	            if tmp_x-1>=0 and tmp_y+1<8 and [tmp_x-1,tmp_y+1] in queens:
	                result.append([tmp_x-1,tmp_y+1])
	                break
	            else:
	                tmp_x -= 1
	                tmp_y += 1
	                
	        tmp_x = x
	        tmp_y = y
	        # diagnal
	        for _ in range(8):
	            if tmp_x+1<8 and tmp_y-1>=0 and [tmp_x+1,tmp_y-1] in queens:
	                result.append([tmp_x+1,tmp_y-1])
	                break
	            else:
	                tmp_x += 1
	                tmp_y -= 1
	        return result
	        
            	      
			
### 运行结果

	Runtime: 28 ms, faster than 67.86% of Python online submissions for Queens That Can Attack the King.
	Memory Usage: 13.4 MB, less than 75.00% of Python online submissions for Queens That Can Attack the King.

### 解析

当然了上面这个解法还是太冗余了，还有代码更加巧妙的解法，核心思路和上面是一样的，从八个不同的方向找第一个出现的皇后的位置记录到结果中，但是代码上简便了很多。

### 解答

	class Solution(object):
	    def queensAttacktheKing(self, queens, king):
	        """
	        :type queens: List[List[int]]
	        :type king: List[int]
	        :rtype: List[List[int]]
	        """
	        result = []
	        for i, j in [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]:
	            x, y = king
	            while 0 <= x < 8 and 0 <= y < 8:
	                x += i
	                y += j
	                if [x, y] in queens:
	                    result.append([x, y])
	                    break
	        return result


### 运行结果

	Runtime: 28 ms, faster than 67.86% of Python online submissions for Queens That Can Attack the King.
	Memory Usage: 13.3 MB, less than 92.86% of Python online submissions for Queens That Can Attack the King.


原题链接：https://leetcode.com/problems/queens-that-can-attack-the-king/



您的支持是我最大的动力
