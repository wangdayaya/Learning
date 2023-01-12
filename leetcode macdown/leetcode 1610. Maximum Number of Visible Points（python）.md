leetcode  1610. Maximum Number of Visible Points（python）

### 描述


You are given an array points, an integer angle, and your location, where location = [pos<sub>x</sub>, pos<sub>y</sub>] and points[i] = [x<sub>i</sub>, y<sub>i</sub>] both denote integral coordinates on the X-Y plane.

Initially, you are facing directly east from your position. You cannot move from your position, but you can rotate. In other words, pos<sub>x</sub> and pos<sub>y</sub> cannot be changed. Your field of view in degrees is represented by angle, determining how wide you can see from any given view direction. Let d be the amount in degrees that you rotate counterclockwise. Then, your field of view is the inclusive range of angles [d - angle/2, d + angle/2].

![](https://assets.leetcode.com/uploads/2020/09/30/angle.mp4)

You can see some set of points if, for each point, the angle formed by the point, your position, and the immediate east direction from your position is in your field of view.

There can be multiple points at one coordinate. There may be points at your location, and you can always see these points regardless of your rotation. Points do not obstruct your vision to other points.

Return the maximum number of points you can see.

Example 1:

![](https://assets.leetcode.com/uploads/2020/09/30/89a07e9b-00ab-4967-976a-c723b2aa8656.png)

	Input: points = [[2,1],[2,2],[3,3]], angle = 90, location = [1,1]
	Output: 3
	Explanation: The shaded region represents your field of view. All points can be made visible in your field of view, including [3,3] even though [2,2] is in front and in the same line of sight.


	
Example 2:

	Input: points = [[2,1],[2,2],[3,4],[1,1]], angle = 90, location = [1,1]
	Output: 4
	Explanation: All points can be made visible in your field of view, including the one at your location.

Example 3:


![](https://assets.leetcode.com/uploads/2020/09/30/5010bfd3-86e6-465f-ac64-e9df941d2e49.png)

	Input: points = [[1,0],[2,1]], angle = 13, location = [1,1]
	Output: 1
	Explanation: You can only see one of the two points, as shown above.
	



Note:


* 1 <= points.length <= 10^5
* points[i].length == 2
* location.length == 2
* 0 <= angle < 360
* 0 <= pos<sub>x</sub>, pos<sub>y</sub>, x<sub>i</sub>, y<sub>i</sub> <= 100

### 解析

根据题意，给定一个列表 points 、一个整数 angle 和您的 location ，其中 location = [pos<sub>x</sub>, pos<sub>y</sub>] 和 points[i] = [x<sub>i</sub>, y<sub>i</sub>] 都表示 X-Y 平面上的积分坐标。

最初，站您的位置直接面向东。你不能从你的位置移动，但你可以旋转。换句话说，pos<sub>x</sub> 和 pos<sub>y</sub> 不能改变。以度为单位的视野由 angle 表示，决定了从任何给定的视线方向可以看到的宽度。让 d 表示您逆时针旋转的度数。然后，您的视野是角度 [d - angle/2, d + angle/2]  的包含范围。

如果对于每个点，由该点、您的位置和从您的位置直接向东形成的角度在您的视野中，您就可以看到一些点集合。一个坐标上可以有多个点。您所在的位置可能有一些点，无论您如何旋转，您始终可以看到这些点。点不会妨碍您对其他点的视线。返回您可以看到的最大点数。

其实这道题读完之后也比较简单，总体思路就是将 location 看成原点，然后计算 points 中的每个点和原点形成的夹角角度，都记录到 angles 中，然后使用滑动窗口思想找在 angle 范围内点数最多的点。题目中有两个点需要格外注意：

* 第一个是计算夹角，需要注意范围为 [0, 2\*pi] ，但是可能在 0 度附近可能上下都有符合题意的角度存在，所以需要将 angles 后再扩充其每个元素加 2\*pi 的值，可以表示 [2\*pi, 4*pi] 的范围
* 第二个就是小数的精度问题，必须要考虑到位，毕竟计算机在计算夹角的时候是浮点数，精度不够可能找点数会出现误差
* 第三个就是有些点就在原点，这些点是始终都能被看到的，直接加到结果即可

### 解答
				
	class Solution(object):
	    def visiblePoints(self, points, angle, location):
	        """
	        :type points: List[List[int]]
	        :type angle: int
	        :type location: List[int]
	        :rtype: int
	        """
	        angles = []
	        pi = 3.1415926
	        origin = 0
	        for x, y in points:
	            dx = x - location[0]
	            dy = y - location[1]
	            if dx == 0 and dy == 0:
	                origin += 1
	                continue
	            alpha = atan2(dy, dx) + pi
	            angles.append(alpha)
	        angles.sort()
	        N = len(angles)
	        for i in range(N):
	            angles.append(angles[i] + 2 * pi)
	
	        result = j = 0
	        for i in range(2*N):
	            while j < 2 * N and angles[j] - angles[i] <= angle * 1.0 * pi / 180 + 0.0000001:
	                j += 1
	            result = max(result, j - i)
	        return result + origin
	        
			
### 运行结果
	Runtime: 2204 ms, faster than 43.70% of Python online submissions for Maximum Number of Visible Points.
	Memory Usage: 56.3 MB, less than 36.97% of Python online submissions for Maximum Number of Visible Points.

原题链接：https://leetcode.com/problems/maximum-number-of-visible-points/



您的支持是我最大的动力
