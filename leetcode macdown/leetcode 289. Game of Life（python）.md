leetcode  289. Game of Life（python）




### 描述


According to Wikipedia's article: "The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970."

The board is made up of an m x n grid of cells, where each cell has an initial state: live (represented by a 1) or dead (represented by a 0). Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) using the following four rules (taken from the above Wikipedia article):

* Any live cell with fewer than two live neighbors dies as if caused by under-population.
* Any live cell with two or three live neighbors lives on to the next generation.
* Any live cell with more than three live neighbors dies, as if by over-population.
* Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.

The next state is created by applying the above rules simultaneously to every cell in the current state, where births and deaths occur simultaneously. Given the current state of the m x n grid board, return the next state.


Example 1:


![](https://assets.leetcode.com/uploads/2020/12/26/grid1.jpg)

	Input: board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
	Output: [[0,0,0],[1,0,1],[0,1,1],[0,1,0]]
	
Example 2:


![](https://assets.leetcode.com/uploads/2020/12/26/grid2.jpg)

	Input: board = [[1,1],[1,0]]
	Output: [[1,1],[1,1]]





Note:

	m == board.length
	n == board[i].length
	1 <= m, n <= 25
	board[i][j] is 0 or 1.


### 解析

根据题意，该面板由 M×N 个细胞组成，其中每个细胞具有初始状态：活着（1）或死亡（0）。每个细胞都使用以下四个规则与其八个邻居（水平，垂直，对角线）相互作用：

* 任何周围少于两个相邻活细胞的活细胞都会死亡
* 任何周围有两个或三个相邻活细胞的活细胞都会生命继续延续
* 任何周围有超过三个相邻活细胞的活细胞都会死亡
* 任何周围有三个相邻活细胞的死细胞都会变成活细胞

通过将上述规则同时应用于当前状态的每个小区，其中出生和死亡同时发生，创建下一个状态。鉴于M X N 网格板的当前状态，返回下一个状态。

这道题目看起来很长，其实很简单，我们只需要掌握住关键的信息即可，对于死细胞来说，周围出现 3 个活细胞的时候会变成活细胞，对于活细胞来说周围活细胞小于 2 或者大于 3 个活细胞就死亡，而且限制条件中 board 的长和宽都很小，所以使用暴力解法对于每个位置上面的细胞做出如上判断进行变化即可。

时间复杂度为 O(M\*N) ，空间复杂度为 O(M\*N)。

很明显上面的解法太简单了，题目难度简直就是 Eazy ，但是这道题目的难度是 Medium ，其实这种解法有一点投机取巧，在原题中还有一点关键的要求，要我们不使用额外的空间来解决题目，直接在 board 上面进行修改，这样会增加难度。

我这里使用到了一种技巧，规则我前面已经讲过了，只要遵循那两种即可，所以我就将所有的元素都变成字符串，然后将他们周围的 1 出现个数统计出来拼接到 board[i][j] 之后，然后重新遍历所有 board[i][j] 按照上面的规则更新每个位置上面的值即可，我是不是一个小聪明【狗头】

时间复杂度为 O(M\*N) ，空间复杂度为 O(1)。


### 解答
				
	class Solution(object):
	    def gameOfLife(self, board):
	        """
	        :type board: List[List[int]]
	        :rtype: None Do not return anything, modify board in-place instead.
	        """
	        N = len(board)
	        M = len(board[0])
	        for i in range(N):
	            for j in range(M):
	                board[i][j] = str(board[i][j])
	        for i in range(N):
	            for j in range(M):
	                count = 0
	                for m in range(max(0, i - 1), min(N, i + 2)):
	                    for n in range(max(0, j - 1), min(M, j + 2)):
	                        if (m != i or n != j) and board[m][n].startswith('1'):
	                            count += 1
	                if count in [2,3]:
	                    board[i][j] += str(count)
	                else:
	                    board[i][j] += str('0')
	        for i in range(N):
	            for j in range(M):
	                if board[i][j].startswith('0') and board[i][j].endswith('3'):
	                    board[i][j] = 1
	                elif board[i][j].startswith('1') and board[i][j].endswith('2') or board[i][j].endswith('3'):
	                    board[i][j] = 1
	                else:
	                    board[i][j] = 0
	    
	    
	            
    
    
            

            	      
			
### 运行结果

	Runtime: 24 ms, faster than 67.36% of Python online submissions for Game of Life.
	Memory Usage: 13.5 MB, less than 40.99% of Python online submissions for Game of Life.



### 原题链接


https://leetcode.com/problems/game-of-life/

您的支持是我最大的动力
