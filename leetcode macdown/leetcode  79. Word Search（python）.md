leetcode 79. Word Search （python）




### 描述

Given an m x n grid of characters board and a string word, return true if word exists in the grid. The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

 



Example 1:

![](https://assets.leetcode.com/uploads/2020/11/04/word2.jpg)

	Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
	Output: true


	
Example 2:

![](https://assets.leetcode.com/uploads/2020/11/04/word-1.jpg)

	Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
	Output: true


Example 3:


![](https://assets.leetcode.com/uploads/2020/10/15/word3.jpg)

	Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
	Output: false


Note:

	m == board.length
	n = board[i].length
	1 <= m, n <= 6
	1 <= word.length <= 15
	board and word consists of only lowercase and uppercase English letters.



### 解析

根据题意，给定一个包含字符和特定单词的 m x n 网格，如果网格中存在单词，则返回 true ，否则返回 false 。该单词可以由顺序相邻的单元格的字母构成，其中相邻单元格指的是水平或垂直相邻。同一个字母单元格不能多次使用。

本题考查的就是深度优先遍历和回溯，为了在 board 中找到可能存在 word ，我们需要进行如下的操作：

遍历所有的 board 中所有的字符，使用递归函数 check 找到一个和 word[0] 相同的字符，说明此时可能是 word 的起点，从这里进行深度优先遍历。

定义递归函数 check (i,j, k) ，表示以 x、y 在 board 的 [x,y] 的位置开始，判断 k+1 在  [x,y] 的上、下、左、右四个方向中相邻的格子中是否存在，如果存在则继续进行递归，当 check 返回为 True 的时候直接跳出循环返回 True 即可，如果不存在则最后返回 False。为了保证每次递归不进行重复操作，我们将已经用过的格子存入 used 中，当经过四个方向的递归之后再从 used 中剔除。

M 、N 为 board 的长和宽，L 为 word 的长度，我们要便利所有的格子，每个格子递归 3^L 次（四个方向中去掉来时的方向不算），所以时间复杂度为 O(M\*N\*3^L) ，由于 used 可能存放 M\*N 个元素，同时递归栈的深度为 min(L，M\*N) ，所以空间复杂度为 O(M\*N)。







### 解答

	class Solution:
	    def exist(self, board, word):
	        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
	        M = len(board)
	        N = len(board[0])
	        def check(i: int, j: int, k: int) -> bool:
	            if board[i][j] != word[k]:
	                return False
	            if k == len(word) - 1:
	                return True
	            used.add((i, j))
	            result = False
	            for delta_i, delta_j in dirs:
	                new_i, new_j = i + delta_i, j + delta_j
	                if 0 <= new_i < M and 0 <= new_j < N:
	                    if (new_i, new_j) not in used:
	                        if check(new_i, new_j, k + 1):
	                            result = True
	                            break
	
	            used.remove((i, j))
	            return result
	        M, N = len(board), len(board[0])
	        used = set()
	        for i in range(M):
	            for j in range(N):
	                if check(i, j, 0):
	                    return True
	        return False
	 
### 运行结果

	Runtime Beats 76.66%
	Memory Beats 52.94%


### 原题链接

https://leetcode.com/problems/word-search/description/


您的支持是我最大的动力
