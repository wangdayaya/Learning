leetcode  36. Valid Sudoku（python）




### 描述

Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

* Each row must contain the digits 1-9 without repetition.
* Each column must contain the digits 1-9 without repetition.
* Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.

Note:

* A Sudoku board (partially filled) could be valid but is not necessarily solvable.
* Only the filled cells need to be validated according to the mentioned rules.



Example 1:

	Input: board = 
	[["5","3",".",".","7",".",".",".","."]
	,["6",".",".","1","9","5",".",".","."]
	,[".","9","8",".",".",".",".","6","."]
	,["8",".",".",".","6",".",".",".","3"]
	,["4",".",".","8",".","3",".",".","1"]
	,["7",".",".",".","2",".",".",".","6"]
	,[".","6",".",".",".",".","2","8","."]
	,[".",".",".","4","1","9",".",".","5"]
	,[".",".",".",".","8",".",".","7","9"]]
	Output: true

	
Example 2:


	Input: board = 
	[["8","3",".",".","7",".",".",".","."]
	,["6",".",".","1","9","5",".",".","."]
	,[".","9","8",".",".",".",".","6","."]
	,["8",".",".",".","6",".",".",".","3"]
	,["4",".",".","8",".","3",".",".","1"]
	,["7",".",".",".","2",".",".",".","6"]
	,[".","6",".",".",".",".","2","8","."]
	,[".",".",".","4","1","9",".",".","5"]
	,[".",".",".",".","8",".",".","7","9"]]
	Output: false
	Explanation: Same as Example 1, except with the 5 in the top left corner being modified to 8. Since there are two 8's in the top left 3x3 sub-box, it is invalid.




Note:

	board.length == 9
	board[i].length == 9
	board[i][j] is a digit 1-9 or '.'.


### 解析

根据题意，让我们判断 9 x 9 大小的数独网格是否有效。数独需要满足以下条件：

* 每行必须包含数字 1-9 ，不得重复
* 每列必须包含数字 1-9 ，不得重复
* 数字 1-9 在每一个以粗实线分隔的 3x3 子网格内只能出现一次。

注意：

* 数独网格不一定可解
* 只有被填充了数字的单元格需要根据上述规则进行验证，空白格子用 '.' 表示，不用判断

本题的逻辑比较清晰，只要同时满足上面三个条件即可，而且数据量较小，网格大小是固定的 9*9 。所以我们使用简单的集合数据结构即可完成判断。思路如下：

* 我们初始化三个不同的字典，第一个字典是用来存储每一行中包含的数字集合，第二个字典是用来存储每一列中包含的数字集合，第三个字典是用来存储每一个 3\*3 子网格中包含的数字集合。
* 我们遍历每一个单元格子 board[i][j]  ，如果 board[i][j]  是空格则直接跳过进入下一个格子的判断；如果 board[i][j]  是数组，且该数字已经存在于 d[0][i] 或者 d[1][j]  或者  d[2][sub_board] 说明其在第 i 行或者第 j 列或者第 sub_board 个子网格已经重复，不符合数独的条件，直接返回 False 即可。否则说明是第一次出现，将数字分别加入  d[0][i] 、 d[1][j]  、  d[2][sub_board]  中。
* 不断重复遍历每一个格子，如果正常结束，就说明都满足三个条件，直接返回 True 即可。

因为遍历所有的格子时间复杂度为 O(1) ，每个格子在三个集合的判断时间也是 O(1)，所以时间复杂度为 O(1) ，因为三个字典的大小都是 9\*9 ，所以空间复杂度为 O(1)。



### 解答

	class Solution:
	    def isValidSudoku(self, board: List[List[str]]) -> bool:
	        d = {0: collections.defaultdict(set),
	            1: collections.defaultdict(set),
	            2: collections.defaultdict(set)}
	        n = 9
	        m = 3
	        for i in range(n):
	            for j in range(n):
	                if board[i][j] == '.':
	                    continue
	                if board[i][j] in d[0][i] or board[i][j] in d[1][j]:
	                    return False
	                sub_board = i // m * m + j // m
	                if board[i][j] in d[2][sub_board]:
	                    return False
	                d[0][i].add(board[i][j])
	                d[1][j].add(board[i][j])
	                d[2][sub_board].add(board[i][j])
	        return True
	 

### 运行结果

	Runtime Beats 91.13%
	Memory Beats 83.37%

### 原题链接


https://leetcode.com/problems/valid-sudoku/description/

您的支持是我最大的动力
