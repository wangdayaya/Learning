leetcode  1041. Robot Bounded In Circle（python）
### 每日经典

《静夜思》 ——李白（唐）

床前明月光，疑是地上霜。

举头望明月，低头思故乡。

### 描述


On an infinite plane, a robot initially stands at (0, 0) and faces north. The robot can receive one of three instructions:

* "G": go straight 1 unit;
* "L": turn 90 degrees to the left;
* "R": turn 90 degrees to the right.
The robot performs the instructions given in order, and repeats them forever.

Return true if and only if there exists a circle in the plane such that the robot never leaves the circle.


Example 1:

	Input: instructions = "GGLLGG"
	Output: true
	Explanation: The robot moves from (0,0) to (0,2), turns 180 degrees, and then returns to (0,0).
	When repeating these instructions, the robot remains in the circle of radius 2 centered at the origin.
	
Example 2:

	Input: instructions = "GG"
	Output: false
	Explanation: The robot moves north indefinitely.

Example 3:

	Input: instructions = "GL"
	Output: true
	Explanation: The robot moves from (0, 0) -> (0, 1) -> (-1, 1) -> (-1, 0) -> (0, 0) -> ...
	


Note:

	1 <= instructions.length <= 100
	instructions[i] is 'G', 'L' or, 'R'.

### 解析

根据题意，在平面上有一个机器人最初站在 (0, 0) 并面向北方。 机器人可以接收以下三种指令之一：

* “G”：直走 1 个单位；
* “L”：向左转 90 度；
* “R”：向右转 90 度。

机器人按顺序执行指令，并永远重复它们。当且仅当平面中存在一个循环的路径使得机器人在执行完 n 遍 instructions 永远能够回到原点并且仍然面向北方返回 True ，否则返回 False 。我们经过找规律可以发现，想要返回 True ，最多执行 4 遍 instructions 之后可能会回到原点并面向北方，如例 3 ；但是也可能最多执行 2 遍 instructions 之后可能回到原点并面向北方，如例 1 。所以我们遍历 4 倍长度的 instructions ，并且对指令数量进行计数，当遇到 G 时位置 cur 发生变化，当遇到 L 或者 R 的时候方向 d 发生变化，如果在遍历过程中满足 count%len(instructions)==0 and d%4==0 and cur==[0,0] 直接返回 True ，否则遍历结束直接返回 False 即可。

### 解答
				
	class Solution(object):
	    def isRobotBounded(self, instructions):
	        """
	        :type instructions: str
	        :rtype: bool
	        """
	        dirs = [(0, 1), (-1, 0), (0, -1), (1, 0)]
	        cur = [0,0]
	        d = 0
	        count = 0
	        for c in instructions*4:
	            count += 1
	            if c == 'G':
	                cur[0] += dirs[d%4][0]
	                cur[1] += dirs[d%4][1]
	            elif c == 'L':
	                d += 1
	            else:
	                d -= 1
	            if count%len(instructions)==0 and d%4==0 and cur==[0,0]:
	                return True
	        return False
	                
            	      
			
### 运行结果

	Runtime: 34 ms, faster than 8.06% of Python online submissions for Robot Bounded In Circle.
	Memory Usage: 13.5 MB, less than 28.85% of Python online submissions for Robot Bounded In Circle.

原题链接：https://leetcode.com/problems/robot-bounded-in-circle/


您的支持是我最大的动力
