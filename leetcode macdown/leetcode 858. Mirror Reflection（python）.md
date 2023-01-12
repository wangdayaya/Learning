leetcode  858. Mirror Reflection（python）




### 描述

There is a special square room with mirrors on each of the four walls. Except for the southwest corner, there are receptors on each of the remaining corners, numbered 0, 1, and 2. The square room has walls of length p and a laser ray from the southwest corner first meets the east wall at a distance q from the 0<sup>th</sup> receptor.

Given the two integers p and q, return the number of the receptor that the ray meets first. The test cases are guaranteed so that the ray will meet a receptor eventually.





Example 1:

![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/06/18/reflection.png)

	Input: p = 2, q = 1
	Output: 2
	Explanation: The ray meets receptor 2 the first time it gets reflected back to the left wall.

	
Example 2:

	Input: p = 3, q = 1
	Output: 1






Note:


	1 <= q <= p <= 1000

### 解析

根据题意，一个特殊的方形房间每面都有镜子。 除西南角外，其余三个角上都有感受器，编号分别为 0、1 和 2 。方形房间有长度为 p 的墙壁，来自西南角的激光射线首先在距离第 0 个接收器 q 处与东墙相遇。给定两个整数 p 和 q ，返回射线首先遇到的感受器的编号。测试用例保证射线最终肯定遇到某个感受器。

因为激光可以在镜子上不断进行反射，所以最后肯定会与某个感受器相碰，所以这道题可以模拟激光的行进路线，来找出最后首次相遇的感受器编号。但是这中做法肯定很绕，我们可以尝试从数学的角度来解决。

光线运动是矢量，所以我们可以拆解为水平和垂直两个方向的运动，水平方向的速度是垂直方向的 p/q 倍，每过一个一个时间单位，光线在水平方向左右两侧来回跳动（移动 p 距离），在垂直方向前进 q 距离，如果到达边界就反方向进行运动，由于感受器的位置在水平方向的两侧，所以只有光线进行整数个时间单位后，才可能达到某个接收器，同理感受器也在垂直方向的两侧，所以只有光线进行 k 个时间单位，也就是总距离为 kq 为 p 的倍数，才会达到某个感受器。

因此只要找到最小整数 k 使得 kq 是 p 的倍数，并且根据 k 的奇偶性可以知道是到达了左侧还是右侧，根据 kq / p 的奇偶性可以得知到达上方还是下方，从综合判断到达的感受器编号。

时间复杂度为 O(logP)，空间复杂度为 O(1) 。



### 解答

	class Solution:
	    def mirrorReflection(self, p: int, q: int) -> int:
	        g = gcd(p, q)
	        p = (p / g) % 2
	        q = (q / g) % 2
	        if p == 1 and q == 1:
	            return 1
	        return 0 if p == 1 else 2

### 运行结果

	Runtime: 34 ms, faster than 90.10% of Python3 online submissions for Mirror Reflection.
	Memory Usage: 13.8 MB, less than 97.03% of Python3 online submissions for Mirror Reflection.

### 原题链接

https://leetcode.com/problems/mirror-reflection/

### 思路参考
https://leetcode.cn/problems/mirror-reflection/solution/jing-mian-fan-she-by-leetcode/


您的支持是我最大的动力
