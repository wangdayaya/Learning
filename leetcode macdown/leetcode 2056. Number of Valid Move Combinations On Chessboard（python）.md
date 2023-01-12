leetcode  2056. Number of Valid Move Combinations On Chessboard（python）

### 描述

There is an 8 x 8 chessboard containing n pieces (rooks, queens, or bishops). You are given a string array pieces of length n, where pieces[i] describes the type (rook, queen, or bishop) of the i<sub>th</sub> piece. In addition, you are given a 2D integer array positions also of length n, where positions[i] = [r<sub>i</sub>, c<sub>i</sub>] indicates that the i<sub>th</sub> piece is currently at the 1-based coordinate (r<sub>i</sub>, c<sub>i</sub>) on the chessboard.

When making a move for a piece, you choose a destination square that the piece will travel toward and stop on.

* A rook can only travel horizontally or vertically from (r, c) to the direction of (r+1, c), (r-1, c), (r, c+1), or (r, c-1).
* A queen can only travel horizontally, vertically, or diagonally from (r, c) to the direction of (r+1, c), (r-1, c), (r, c+1), (r, c-1), (r+1, c+1), (r+1, c-1), (r-1, c+1), (r-1, c-1).
* A bishop can only travel diagonally from (r, c) to the direction of (r+1, c+1), (r+1, c-1), (r-1, c+1), (r-1, c-1).

You must make a move for every piece on the board simultaneously. A move combination consists of all the moves performed on all the given pieces. Every second, each piece will instantaneously travel one square towards their destination if they are not already at it. All pieces start traveling at the 0<sub>th</sub> second. A move combination is invalid if, at a given time, two or more pieces occupy the same square.

Return the number of valid move combinations​​​​​.

Notes:

* No two pieces will start in the same square.
* You may choose the square a piece is already on as its destination.
* If two pieces are directly adjacent to each other, it is valid for them to move past each other and swap positions in one second.
 



Example 1:

![](https://assets.leetcode.com/uploads/2021/09/23/a1.png)

	Input: pieces = ["rook"], positions = [[1,1]]
	Output: 15
	Explanation: The image above shows the possible squares the piece can move to.


	
Example 2:

![](https://assets.leetcode.com/uploads/2021/09/23/a2.png)

	Input: pieces = ["queen"], positions = [[1,1]]
	Output: 22
	Explanation: The image above shows the possible squares the piece can move to.



Example 3:


![](https://assets.leetcode.com/uploads/2021/09/23/a3.png)
	
	Input: pieces = ["bishop"], positions = [[4,3]]
	Output: 12
	Explanation: The image above shows the possible squares the piece can move to.
	
Example 4:

![](https://assets.leetcode.com/uploads/2021/09/23/a4.png)

	Input: pieces = ["rook","rook"], positions = [[1,1],[8,8]]
	Output: 223
	Explanation: There are 15 moves for each rook which results in 15 * 15 = 225 move combinations.
	However, there are two invalid move combinations:
	- Move both rooks to (8, 1), where they collide.
	- Move both rooks to (1, 8), where they collide.
	Thus there are 225 - 2 = 223 valid move combinations.
	Note that there are two valid move combinations that would result in one rook at (1, 8) and the other at (8, 1).
	Even though the board state is the same, these two move combinations are considered different since the moves themselves are different.

	
Example 5:


![](https://assets.leetcode.com/uploads/2021/09/23/a5.png)

	Input: pieces = ["queen","bishop"], positions = [[5,7],[3,4]]
	Output: 281
	Explanation: There are 12 * 24 = 288 move combinations.
	However, there are several invalid move combinations:
	- If the queen stops at (6, 7), it blocks the bishop from moving to (6, 7) or (7, 8).
	- If the queen stops at (5, 6), it blocks the bishop from moving to (5, 6), (6, 7), or (7, 8).
	- If the bishop stops at (5, 2), it blocks the queen from moving to (5, 2) or (5, 1).
	Of the 288 move combinations, 281 are valid.

Note:

* n == pieces.length
* n == positions.length
* 1 <= n <= 4
* pieces only contains the strings "rook", "queen", and "bishop".
* There will be at most one queen on the chessboard.
* 1 <= x<sub>i</sub>, y<sub>i</sub> <= 8
* Each positions[i] is distinct.



### 解析


根据题意，有一个 8 x 8 的棋盘，包含 n 个棋子（车、皇后或主教）。给定一个长度为 n 的字符串数组 pieces，其中 pieces[i] 描述了第 i 个片段的类型（车、王后或主教）。此外给出一个长度为 n 的二维整数数组 positions ，其中 positions[i] = [r<sub>i</sub>, c<sub>i</sub>] 表示第 i 个棋子当前位于棋盘上的从 1 开始的的坐标 (r<sub>i</sub>, c<sub>i</sub>) 上。

当棋子移动时，可以选择棋子将行进并停在的目标方格：

* 车只能从（r，c）向（r+1，c）、（r-1，c）、（r，c+1）或（r，c-1）的方向水平或垂直移动
* 皇后只能水平、垂直或斜行从 (r, c) 到 (r+1, c) 、(r-1, c) 、(r, c+1) 、(r, c- 1) 、 (r+1, c+1) 、(r+1, c-1) 、(r-1, c+1) 、(r-1, c-1)
* 主教只能从 (r, c) 向 (r+1, c+1) 、(r+1, c-1) 、 (r-1, c+1) 、(r-1,  c-1) 斜行

同时对棋盘上的每个棋子进行移动。所有棋子从第 0 秒开始行进，所有的棋子的行进结束位置点一共有多少种组合。如果在给定时间，两个或多个棋子占据同一方格，则移动组合无效。返回有效的移动组合数。

注意：

* 没有两个棋子会在同一个方块中开始
* 可以选择棋子所在的方格作为目的地，也就是可以不走
* 如果两个棋子彼此直接相邻，则它们可以在一秒钟内相互移动并交换位置

其实这道题的英文题意不是很清楚，主要是说了有三种棋子，每种棋子的行进方法，棋子的停止有两种情况：出界和碰撞。然后告诉我们移动（也可以不移动）棋子之后，棋盘上的棋面有多少种组合。

查看限制条件发现最多 4 个棋子，且最多只有一个皇后，那么最多 4 个棋子某个时刻的方向组合有 4\*4\*4\*8=256 种方向选择，而且棋盘也很小，每个棋子的行进位置也有限，像这种题很明显大方向是要用 DFS 来暴力遍历所有棋子的所有可能位置，找出所有可能出现的棋面组合。关键在于一些小的技巧。

如果有 N 个棋子，因为只有皇后的可能行进方向最多，有 8 个方向，使用 3\*N 位 bit 来表示所有棋子的状态，每 3 位表示一个棋子的状态。另外还可以用 hash 的方法来保存已经出现的棋面组合，这样可以减少很多不必要的递归。

### 解答
				
	
	class Solution(object):
	    def __init__(self):
	        self.result = set()
	        self.dir = [[-1, 0], [1, 0], [0, -1], [0, 1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
	        self.N = 0
	
	    def countCombinations(self, pieces, positions):
	        """
	        :type pieces: List[str]
	        :type positions: List[List[int]]
	        :rtype: int
	        """
	        self.N = len(pieces)
	        for state in range(1 << (3 * self.N)):
	            flag = 1
	            dirs = [0] * self.N
	            for k in range(self.N):
	                d = (state >> (3 * k)) % 8
	                if pieces[k] == 'rook' and d > 3:
	                    flag = 0
	                    break
	                if pieces[k] == 'bishop' and d < 4:
	                    flag = 0
	                    break
	                dirs[k] = d
	            if flag:
	                self.dfs(positions, dirs, (1 << self.N) - 1)
	        return len(self.result) + 1
	
	    def dfs(self, positions, dirs, state):
	        subset = state
	        while subset>0:
	            pos2 = copy.deepcopy(positions)
	            flag = 1
	            for i in range(self.N):
	                if (subset >> i) & 1:
	                    pos2[i][0] += self.dir[dirs[i]][0]
	                    pos2[i][1] += self.dir[dirs[i]][1]
	                    if pos2[i][0] < 1 or pos2[i][0] > 8:
	                        flag = 0
	                        break
	                    if pos2[i][1] < 1 or pos2[i][1] > 8:
	                        flag = 0
	                        break
	
	            if flag == 0:
	                subset = (subset - 1) & state
	                continue
	            if self.duplicate(pos2):
	                subset = (subset - 1) & state
	                continue
	
	            hash = self.getMyHash(pos2)
	            if hash in self.result:
	                subset = (subset - 1) & state
	                continue
	            self.result.add(hash)
	            self.dfs(pos2, dirs, subset)
	            subset = (subset - 1) & state
	
	
	    def duplicate(self, pos):
	        for i in range(self.N):
	            for j in range(i + 1, self.N):
	                if pos[i] == pos[j]:
	                    return True
	        return False
	
	    def getMyHash(self, pos):
	        result = 0
	        for i in range(self.N):
	            result = result * 10 + pos[i][0]
	            result = result * 10 + pos[i][1]
	        return result
            	      
			
### 运行结果

	
	Runtime: 5016 ms, faster than 27.12% of Python online submissions for Number of Valid Move Combinations On Chessboard.
	Memory Usage: 17.7 MB, less than 44.07% of Python online submissions for Number of Valid Move Combinations On Chessboard.


原题链接：https://leetcode.com/problems/number-of-valid-move-combinations-on-chessboard/



您的支持是我最大的动力
