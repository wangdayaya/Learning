leetcode  2146. K Highest Ranked Items Within a Price Range（python）

### 前言



刚刚过去的周末参加了 Biweekly Contest 70  ，我只做出来三道题，真的是惭愧。这是 Biweekly Contest 70 的第三题，难度 Medium ，其实就考察 BFS ，只要理清思路就基本上能做出来。


### 描述


You are given a 0-indexed 2D integer array grid of size m x n that represents a map of the items in a shop. The integers in the grid represent the following:

* 0 represents a wall that you cannot pass through.
* 1 represents an empty cell that you can freely move to and from.
* All other positive integers represent the price of an item in that cell. You may also freely move to and from these item cells.

It takes 1 step to travel between adjacent grid cells.You are also given integer arrays pricing and start where pricing = [low, high] and start = [row, col] indicates that you start at the position (row, col) and are interested only in items with a price in the range of [low, high] . You are further given an integer k.

You are interested in the positions of the k highest-ranked items whose prices are within the given price range. The rank is determined by the first of these criteria that is different:

* Distance, defined as the length of the shortest path from the start (shorter distance has a higher rank).
* Price (lower price has a higher rank, but it must be in the price range).
* The row number (smaller row number has a higher rank).
* The column number (smaller column number has a higher rank).

Return the k highest-ranked items within the price range sorted by their rank (highest to lowest). If there are fewer than k reachable items within the price range, return all of them.


Example 1:

![](https://assets.leetcode.com/uploads/2021/12/16/example1drawio.png)

	Input: grid = [[1,2,0,1],[1,3,0,1],[0,2,5,1]], pricing = [2,5], start = [0,0], k = 3
	Output: [[0,1],[1,1],[2,1]]
	Explanation: You start at (0,0).
	With a price range of [2,5], we can take items from (0,1), (1,1), (2,1) and (2,2).
	The ranks of these items are:
	- (0,1) with distance 1
	- (1,1) with distance 2
	- (2,1) with distance 3
	- (2,2) with distance 4
	Thus, the 3 highest ranked items in the price range are (0,1), (1,1), and (2,1).


Note:

	m == grid.length
	n == grid[i].length
	1 <= m, n <= 10^5
	1 <= m * n <= 10^5
	0 <= grid[i][j] <= 10^5
	pricing.length == 2
	2 <= low <= high <= 10^5
	start.length == 2
	0 <= row <= m - 1
	0 <= col <= n - 1
	grid[row][col] > 0
	1 <= k <= m * n


### 解析


根据题意，给定一个 0 索引的 2D 整数数组 grid ，大小为 m x n，表示商店中商品的地图。网格中的整数表示以下内容：

* 0 代表你不能穿过的墙
* 1 代表一个可以自由移动的空单元格
* 其他正整数表示该单元格中物品的价格，人可以自由地进出这些项目单元格。

在相邻的网格单元之间移动需要 1 步。题目还给出了整数数组 pricing 和  start ，其中pricing = [low, high] 和 start = [row, col] 表示人会从位置 (row, col) 开始走动并购物，并且只对 pricing 范围内的项目感兴趣，最后还给出一个整数 k 。让我们返回价格在给定价格范围内的排名最高的 k 个商品的位置，少于 k 个，则全部返回。排名由以下四个标准中的较高优先级决定：

* 距离，定义为从起点到最短路径的长度（距离越短，等级越高）
* 价格（价格越低排名越高，但必须在价格范围内）
* 行号（行号越小排名越高）
* 列号（列号越小排名越高）

这道题还是很贴近日常的生活的，我们在购物的时候其实也能用这个题目来计算自己在超市的购物位置，当然了需要考虑的标准你可以根据自己的来。言归正传，读完这道题一看就知道是要使用 BFS 的思路：

我们需要定义一个 stack 用来保存经过的位置，我们将栈中的元素都定义为 [价格，横坐标，纵坐标] 的形式，一开始里面只有起始位置和商品价格 [grid[start[0]][start[1]], start[0], start[1]]  ；同时定义一个集合 visted 来存放已经走过的位置，保证不会重复购物，如果一开始的位置就在意向价格范围内，那我们就将其位置加入结果 result ，此时如果 k 为 1 ，那么直接返回 result 即可。

然后当 stack 不为空的时候进行 while 循环，我没有把距离加入到栈元素的形式，就是因为在进行栈遍历的时候就是按照距离来进行的，这是基于了 BFS 的特性，从 start 位置开始使用 BFS 的方式进行下一个位置的查找，肯定是会先把最近的距离都过一次，才会去找较远的距离，你可以看成在同一距离的商品在一个类似等高线上，我们这里使用了 tmp 列表来存放当前 stack 中元素即将进入下一个位置的所有商品的信息，这些在 tmp 中商品的距离都相等，然后对 tmp 进行排序，因为里面有三个元素都是越小优先级越高，所以使用内置函数 sort 就可以直接升序排序即可。然后遍历排序之后的 tmp ，只要价格在意向价格之内就将其位置加入到 result 中，如果 result 已经有 k 个，直接返回即可。

while 循环结束之后说明能选的意向商品不足 k 个，直接将所有结果 result 返回即可。

这是一开始想到的原汁原味的代码，所以直接贴了上来，思路就是以上全部，代码有点乱，其实还可以优化一下，时间复杂度是 O(n) ，空间复杂度是 O(n) 。
### 解答
				

	class Solution(object):
	    def highestRankedKItems(self, grid, pricing, start, k):
	        """
	        :type grid: List[List[int]]
	        :type pricing: List[int]
	        :type start: List[int]
	        :type k: int
	        :rtype: List[List[int]]
	        """
	        M = len(grid)
	        N = len(grid[0])
	        L = pricing[0]
	        R = pricing[1]
	        dirs = [[-1,0],[0,-1],[0,1],[1,0]]
	        visted = set()
	        visted.add((start[0], start[1]))
	        stack = [[grid[start[0]][start[1]], start[0], start[1]]]
	        result = []
	        if L<=grid[start[0]][start[1]]<=R:
	            result.append([start[0],start[1]])
	        if len(result) == k:
	            return result
	        while stack:
	            tmp = []
	            num = len(stack)
	            while num!=0:
	                p, x, y = stack.pop(0)
	                for d in dirs:
	                    t_x = x + d[0]
	                    t_y = y + d[1]
	                    if t_x<0 or t_x>=M or t_y<0 or t_y>=N: continue
	                    if (t_x, t_y) in visted: continue
	                    if grid[t_x][t_y] == 0: continue
	                    visted.add((t_x, t_y))
	                    tmp.append([grid[t_x][t_y], t_x, t_y])
	                num -= 1
	            tmp.sort()
	            stack.extend(tmp)
	            for item in tmp:
	                if L<=item[0]<=R:
	                    result.append([item[1], item[2]])
	                    if len(result) == k:
	                        return result
	        return result
	        
	        
            	      
			
### 运行结果

	Runtime: 3284 ms, faster than 100.00% of Python online submissions for K Highest Ranked Items Within a Price Range.
	Memory Usage: 62.2 MB, less than 100.00% of Python online submissions for K Highest Ranked Items Within a Price Range.

### 原题链接

https://leetcode.com/contest/biweekly-contest-70/problems/k-highest-ranked-items-within-a-price-range/




您的支持是我最大的动力
