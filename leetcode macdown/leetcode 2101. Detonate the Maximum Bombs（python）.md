leetcode  2101. Detonate the Maximum Bombs（python）

### 描述

You are given a list of bombs. The range of a bomb is defined as the area where its effect can be felt. This area is in the shape of a circle with the center as the location of the bomb.

The bombs are represented by a 0-indexed 2D integer array bombs where bombs[i] = [x<sub>i</sub>, y<sub>i</sub>, r<sub>i</sub>]. x<sub>i</sub> and y<sub>i</sub> denote the X-coordinate and Y-coordinate of the location of the i<sub>th</sub> bomb, whereas r<sub>i</sub> denotes the radius of its range.

You may choose to detonate a single bomb. When a bomb is detonated, it will detonate all bombs that lie in its range. These bombs will further detonate the bombs that lie in their ranges.

Given the list of bombs, return the maximum number of bombs that can be detonated if you are allowed to detonate only one bomb.



Example 1:

![](https://assets.leetcode.com/uploads/2021/11/06/desmos-eg-3.png)

	Input: bombs = [[2,1,3],[6,1,4]]
	Output: 2
	Explanation:
	The above figure shows the positions and ranges of the 2 bombs.
	If we detonate the left bomb, the right bomb will not be affected.
	But if we detonate the right bomb, both bombs will be detonated.
	So the maximum bombs that can be detonated is max(1, 2) = 2.

	
Example 2:


![](https://assets.leetcode.com/uploads/2021/11/06/desmos-eg-2.png)

	Input: bombs = [[1,1,5],[10,10,5]]
	Output: 1
	Explanation:
	Detonating either bomb will not detonate the other bomb, so the maximum number of bombs that can be detonated is 1.

Example 3:

![](https://assets.leetcode.com/uploads/2021/11/07/desmos-eg1.png)

	Input: bombs = [[1,2,3],[2,3,1],[3,4,2],[4,5,3],[5,6,4]]
	Output: 5
	Explanation:
	The best bomb to detonate is bomb 0 because:
	- Bomb 0 detonates bombs 1 and 2. The red circle denotes the range of bomb 0.
	- Bomb 2 detonates bomb 3. The blue circle denotes the range of bomb 2.
	- Bomb 3 detonates bomb 4. The green circle denotes the range of bomb 3.
	Thus all 5 bombs are detonated.




Note:

* 	1 <= bombs.length <= 100
* 	bombs[i].length == 3
* 	1 <= x<sub>i</sub>, y<sub>i</sub>, r<sub>i</sub> <= 10^5


### 解析


根据题意，给定一份炸弹清单。 炸弹的射程定义为可以感觉到其影响的区域。 该区域呈圆形，以炸弹的位置为中心。炸弹由 0 开始索引的 2D 整数数组 bombs 表示，其中bombs[i] = [x<sub>i</sub>, y<sub>i</sub>, r<sub>i</sub>] 。 x<sub>i</sub>和 y<sub>i</sub> 表示第 i 个炸弹位置的 X 坐标和 Y 坐标，而 r<sub>i</sub> 表示其射程半径。可以选择引爆单个炸弹。 当炸弹被引爆时，它将引爆其范围内的所有炸弹。 这些炸弹将进一步引爆位于其射程内的炸弹。给定炸弹列表，如果只允许引爆一枚炸弹，则返回可以引爆的最大炸弹数量。

其实看了限制条件，炸弹的数量最多有 100 个，所以可以使用暴力的方法，将所有的点作为其实爆炸点遍历一遍，找出可以引爆的最大炸弹数量，这个过程需要 O(n) ，另外在选中一个爆炸点之后，可以维护一个队列，时间复杂度为 O(n) ，使用 BFS 将所有的后续爆炸点都找出来，这个过程需要 O(n) ，因为每个爆炸点可能最多引爆 n 个炸弹，所以总共需要 O(n^3) ，勉强可以通过。

### 解答
				

	class Solution(object):
	    def maximumDetonation(self, bombs):
	        N = len(bombs)
	        nxt = [[] for _ in range(N)]
	        for i in range(N):
	            for j in range(N):
	                if i == j: continue
	                dx = bombs[i][0]-bombs[j][0]
	                dy = bombs[i][1]-bombs[j][1]
	                if dx*dx + dy*dy <= bombs[i][2]*bombs[i][2]:
	                    nxt[i].append(j)
	        result = 0
	        for start in range(N):
	            queue = []
	            queue.append(start)
	            visited = [0] * N
	            visited[start] = 1 
	            while queue:
	                for bomb in nxt[queue.pop(0)]:
	                    if visited[bomb]:
	                        continue
	                    queue.append(bomb)
	                    visited[bomb] = 1
	            result = max(result, sum(visited))
	        return result                    
	            
            	      
			
### 运行结果

	Runtime: 724 ms, faster than 60.87% of Python online submissions for Detonate the Maximum Bombs.
	Memory Usage: 13.8 MB, less than 65.22% of Python online submissions for Detonate the Maximum Bombs.


原题链接：https://leetcode.com/problems/detonate-the-maximum-bombs/submissions/

您的支持是我最大的动力
