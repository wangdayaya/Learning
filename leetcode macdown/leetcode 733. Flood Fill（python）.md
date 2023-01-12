leetcode  733. Flood Fill（python）

### 描述


An image is represented by an m x n integer grid image where image[i][j] represents the pixel value of the image.

You are also given three integers sr, sc, and newColor. You should perform a flood fill on the image starting from the pixel image[sr][sc].

To perform a flood fill, consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color), and so on. Replace the color of all of the aforementioned pixels with newColor.

Return the modified image after performing the flood fill.


Example 1:

![](https://assets.leetcode.com/uploads/2021/06/01/flood1-grid.jpg)

	Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, newColor = 2
	Output: [[2,2,2],[2,2,0],[2,0,1]]
	Explanation: From the center of the image with position (sr, sc) = (1, 1) (i.e., the red pixel), all pixels connected by a path of the same color as the starting pixel (i.e., the blue pixels) are colored with the new color.
	Note the bottom corner is not colored 2, because it is not 4-directionally connected to the starting pixel.

	
Example 2:

	Input: image = [[0,0,0],[0,0,0]], sr = 0, sc = 0, newColor = 2
	Output: [[2,2,2],[2,2,2]]





Note:

	m == image.length
	n == image[i].length
	1 <= m, n <= 50
	0 <= image[i][j], newColor < 2^16
	0 <= sr < m
	0 <= sc < n


### 解析

根据题意，给出了一个 MxN 的照片 image , image[i][j] 的值代表着像素。题目还给出了起点的位置 [sr,sc] 和一个待用的颜色像素值 newColor ，题目要求我们从照片的  image[sr][sc] 开始执行 flood fill 操作。

flood fill 操作其实就是如果该位置的上下左右四个相邻位置或者其上下左右的四个相邻位置的上下左右的位置（不断递归的位置）都与 image[sr][sc]  的值相同，就用 newColor 来替换他们的值。思路比较简单，类似于 BFS ：

* 初始化 M 表示行数， N 表示列数，visited 表示最后待转换值的位置， oldColor 记录起始位置的像素，stack 用来记录经过的所有的上下左右的有效位置
* 遍历 stack 中的位置，弹出 stack 的第一个位置，将其加入到 visited 中，然后找该位置相邻四个方向的有效位置，继续循环遍历
* 最后遍历 visited 中的所有位置，用 newColor 替换这些位置上旧的值，最后返回 image 即可


### 解答
				

	class Solution(object):
	    def floodFill(self, image, sr, sc, newColor):
	        """
	        :type image: List[List[int]]
	        :type sr: int
	        :type sc: int
	        :type newColor: int
	        :rtype: List[List[int]]
	        """
	        M = len(image)
	        N = len(image[0])
	        stack = [[sr, sc]]
	        visited = []
	        oldColor = image[sr][sc]
	        while stack:
	            x, y = stack.pop(0)
	            visited.append([x, y])
	            if x + 1 < M and image[x + 1][y] == oldColor:
	                if [x + 1, y] not in visited:
	                    stack.append([x + 1, y])
	            if x - 1 > -1 and image[x - 1][y] == oldColor:
	                if [x - 1, y] not in visited:
	                    stack.append([x - 1, y])
	            if y - 1 > -1 and image[x][y - 1] == oldColor:
	                if [x, y - 1] not in visited:
	                    stack.append([x, y - 1])
	            if y + 1 < N and image[x][y + 1] == oldColor:
	                if [x, y + 1] not in visited:
	                    stack.append([x, y + 1])
	        for x, y in visited:
	            image[x][y] = newColor
	        return image
            	      
			
### 运行结果

	Runtime: 52 ms, faster than 97.83% of Python online submissions for Flood Fill.
	Memory Usage: 13.3 MB, less than 93.73% of Python online submissions for Flood Fill.


### 解析

当然还可以用 DFS 来解题，原理和上面类似。看运行结果好像不如上面的 BFS 好。

### 解答
			
	class Solution(object):
	    def floodFill(self, image, sr, sc, newColor):
	        """
	        :type image: List[List[int]]
	        :type sr: int
	        :type sc: int
	        :type newColor: int
	        :rtype: List[List[int]]
	        """
	        M = len(image)
	        N = len(image[0])
	        oldColor = image[sr][sc]
	        if oldColor == newColor:return image
	        def dfs(x, y):
	            if image[x][y] == oldColor:
	                image[x][y] = newColor
	                if x-1 > -1: dfs(x-1, y)
	                if x+1 < M: dfs(x+1, y)
	                if y-1 > -1: dfs(x, y-1)
	                if y+1 < N: dfs(x, y+1)
	        dfs(sr, sc)
	        return image			

### 运行结果

	Runtime: 56 ms, faster than 89.39% of Python online submissions for Flood Fill.
	Memory Usage: 13.6 MB, less than 41.22% of Python online submissions for Flood Fill.


原题链接：https://leetcode.com/problems/flood-fill/



您的支持是我最大的动力
