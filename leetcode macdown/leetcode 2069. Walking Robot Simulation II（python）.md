leetcode  2069. Walking Robot Simulation II（python）

### 描述


A width x height grid is on an XY-plane with the bottom-left cell at (0, 0) and the top-right cell at (width - 1, height - 1). The grid is aligned with the four cardinal directions ("North", "East", "South", and "West"). A robot is initially at cell (0, 0) facing direction "East".

The robot can be instructed to move for a specific number of steps. For each step, it does the following.

* Attempts to move forward one cell in the direction it is facing.
* If the cell the robot is moving to is out of bounds, the robot instead turns 90 degrees counterclockwise and retries the step.

After the robot finishes moving the number of steps required, it stops and awaits the next instruction.

Implement the Robot class:

* Robot(int width, int height) Initializes the width x height grid with the robot at (0, 0) facing "East".
* void step(int num) Instructs the robot to move forward num steps.
* int[] getPos() Returns the current cell the robot is at, as an array of length 2, [x, y].
* String getDir() Returns the current direction of the robot, "North", "East", "South", or "West".


Example 1:


![](https://assets.leetcode.com/uploads/2021/10/09/example-1.png)

	Input
	["Robot", "move", "move", "getPos", "getDir", "move", "move", "move", "getPos", "getDir"]
	[[6, 3], [2], [2], [], [], [2], [1], [4], [], []]
	Output
	[null, null, null, [4, 0], "East", null, null, null, [1, 2], "West"]
	
	Explanation
	Robot robot = new Robot(6, 3); // Initialize the grid and the robot at (0, 0) facing East.
	robot.move(2);  // It moves two steps East to (2, 0), and faces East.
	robot.move(2);  // It moves two steps East to (4, 0), and faces East.
	robot.getPos(); // return [4, 0]
	robot.getDir(); // return "East"
	robot.move(2);  // It moves one step East to (5, 0), and faces East.
	                // Moving the next step East would be out of bounds, so it turns and faces North.
	                // Then, it moves one step North to (5, 1), and faces North.
	robot.move(1);  // It moves one step North to (5, 2), and faces North (not West).
	robot.move(4);  // Moving the next step North would be out of bounds, so it turns and faces West.
	                // Then, it moves four steps West to (1, 2), and faces West.
	robot.getPos(); // return [1, 2]
	robot.getDir(); // return "West"

	



Note:

	2 <= width, height <= 100
	1 <= num <= 10^5
	At most 10^4 calls in total will be made to step, getPos, and getDir.


### 解析


根据题意，在一个 width x height 的网格在一个坐标系上，左下角的单元格位于 (0, 0)，右上角的单元格位于 (width - 1, height - 1)。网格与四个基本方向 “北”、“东”、“南”和“西” 对齐。机器人最初在单元格 (0, 0) 处面向“东”方向。可以指示机器人移动，但是有以下的限制条件：

* 可以沿它所面对的方向向前移动若干个指定数目的单元格
* 如果机器人移动到的单元格越界，则机器人会逆时针旋转 90 度并继续向前移动

机器人完成移动所需的步数后，它会停止并等待下一条指令。题目要求实现 Robot 类中的函数：

* Robot(int width, int height) ：初始化宽度 x 高度网格，机器人在 (0, 0) 处面向“东”
* void step(int num) ：指示机器人向前移动 num 步
* int[] getPos() ：以 [x, y] 的数组形式返回机器人所在的当前单元格位置
* String getDir() ：返回机器人的当前面向方向，“北”、“东”、“南”或“西”

其实读完题之后，基本对题意也比较了解了，很多人都会直接上来就初始化一个 width x height 的网格，然后按照题意写机器人的移动规律代码，我一开始就是这么想的，但是当我写了几行代码突然发现了一个规律，那就是这个机器人只在最外面的一圈做一个逆时针的运动，中间的格子从来不会被走过，所以这么一来题目就简单多了。

我们只需要将最外层的一圈格子的位置，拉成一条首尾相接的圆形跑道，将格子的位置和当前格子的面朝方向存入一个列表中，然后让机器人不断地走，直接将走的总步数对列表长度取余即可找出所在格子位置，但是需要注意的是起始位置的 (0,0) 的方向，除了一开始面向 East ，其他时候都是面向 South 的，真的吐血，提交了三次 Error ，怪不得这道题通过率这么低。

### 解答
				

	class Robot(object):
	    def __init__(self, width, height):
	        """
	        :type width: int
	        :type height: int
	        """
	        self.path = [[x,0,'East'] for x in range(width)] + [[width-1, y, 'North'] for y in range(1,height)] + [[x, height-1, 'West'] for x in range(width-2, -1, -1)] + [[0, y, 'South'] for y in range(height-2, 0, -1)]
	        self.path[0][2] = 'South'
	        self.count = 0
	        
	        
	    def step(self, num):
	        """
	        :type num: int
	        :rtype: None
	        """
	        self.count += num
	        
	        
	
	    def getPos(self):
	        """
	        :rtype: List[int]
	        """
	        return self.path[self.count%len(self.path)][:2]
	        
	
	    def getDir(self):
	        """
	        :rtype: str
	        """
	        return self.path[self.count%len(self.path)][2] if self.count!=0 else 'East'
            	      
			
### 运行结果

	Runtime: 332 ms, faster than 88.89% of Python online submissions for Walking Robot Simulation II.
	Memory Usage: 18.4 MB, less than 20.83% of Python online submissions for Walking Robot Simulation II.


原题链接：https://leetcode.com/problems/walking-robot-simulation-ii/



您的支持是我最大的动力
