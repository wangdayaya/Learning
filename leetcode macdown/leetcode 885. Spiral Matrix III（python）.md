leetcode  885. Spiral Matrix III（python）

### 描述


You start at the cell (rStart, cStart) of an rows x cols grid facing east. The northwest corner is at the first row and column in the grid, and the southeast corner is at the last row and column.

You will walk in a clockwise spiral shape to visit every position in this grid. Whenever you move outside the grid's boundary, we continue our walk outside the grid (but may return to the grid boundary later.). Eventually, we reach all rows * cols spaces of the grid.

Return an array of coordinates representing the positions of the grid in the order you visited them.

 


Example 1:
![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a329a1de29ee4445b40d9b8a419fdd98~tplv-k3u1fbpfcp-zoom-1.image)

	Input: rows = 1, cols = 4, rStart = 0, cStart = 0
	Output: [[0,0],[0,1],[0,2],[0,3]]
	
	
Example 2:

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/9a497d31167d4601ab6f775d69bf8d83~tplv-k3u1fbpfcp-zoom-1.image)

	Input: rows = 5, cols = 6, rStart = 1, cStart = 4
	Output: [[1,4],[1,5],[2,5],[2,4],[2,3],[1,3],[0,3],[0,4],[0,5],[3,5],[3,4],[3,3],[3,2],[2,2],[1,2],[0,2],[4,5],[4,4],[4,3],[4,2],[4,1],[3,1],[2,1],[1,1],[0,1],[4,0],[3,0],[2,0],[1,0],[0,0]]



Note:

	1 <= rows, cols <= 100
	0 <= rStart < rows
	0 <= cStart < cols


### 解析

根据题意，就是给出了一个结果二维列表的行数和列数，并且告诉起始位置在 (x,y) ，从起始位置开始按顺时针方向以螺旋方向行走，并且经过的位置的数字从 1 开始递增，最后让我们返回行走过程中走过的位置列表。

其实这个题的思路比较简单，就是按照题意来按照 “右下左上右” 的顺序不断的移动位置，将经过的位置存放到结果列表 result 中，注意的是超出列表范围仍然可以移动，但是位置却不能保存在 result 中。

### 解答
				
	class Solution(object):
	    def spiralMatrixIII(self,  rows, cols, x, y):
	        """
	        :type rows: int
	        :type cols: int
	        :type x: int
	        :type y: int
	        :rtype: List[List[int]]
	        """
	        result = [(x,y)]
	        circle = 1
	        while True:
	            y += 1
	            if x >= 0 and x < rows and y >= 0 and y < cols:
	                result.append((x,y))
	            if len(result) == rows * cols: return result
	            # 向下
	            for _ in range(circle * 2 - 1):
	                x += 1
	                if x>=0 and x<rows and y >= 0 and y<cols:
	                    result.append((x,y))
	                if len(result) == rows * cols: return result
	            # 向左
	            for _ in range(2 * circle):
	                y -= 1
	                if x>=0 and x<rows and y >= 0 and y<cols :
	                    result.append((x,y))
	                if len(result) == rows * cols: return result
	            # 向上
	            for _ in range(2 * circle):
	                x -= 1
	                if x>=0 and x<rows and y >= 0 and y<cols:
	                    result.append((x,y))
	                if len(result) == rows * cols: return result
	            # 向右
	            for _ in range(2 * circle):
	                y += 1
	                if x>=0 and x<rows and y >= 0 and y<cols:
	                    result.append((x,y))
	                if len(result) == rows * cols: return result
	            circle += 1
	            
	        
	        	            
	            
	        
            	      
			
### 运行结果

	Runtime: 161 ms, faster than 28.13% of Python online submissions for Spiral Matrix III.
	Memory Usage: 14.3 MB, less than 96.88% of Python online submissions for Spiral Matrix III.

原题链接：https://leetcode.com/problems/spiral-matrix-iii


您的支持是我最大的动力