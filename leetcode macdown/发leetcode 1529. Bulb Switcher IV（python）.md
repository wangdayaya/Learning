leetcode  1529. Bulb Switcher IV（python）

### 描述


There is a room with n bulbs, numbered from 0 to n - 1, arranged in a row from left to right. Initially, all the bulbs are turned off.

Your task is to obtain the configuration represented by target where target[i] is '1' if the ith bulb is turned on and is '0' if it is turned off.

You have a switch to flip the state of the bulb, a flip operation is defined as follows:

* Choose any bulb (index i) of your current configuration.
* Flip each bulb from index i to index n - 1.

When any bulb is flipped it means that if it is '0' it changes to '1' and if it is '1' it changes to '0'.

Return the minimum number of flips required to form target.




Example 1:


	Input: target = "10111"
	Output: 3
	Explanation: Initial configuration "00000".
	flip from the third bulb:  "00000" -> "00111"
	flip from the first bulb:  "00111" -> "11000"
	flip from the second bulb:  "11000" -> "10111"
	We need at least 3 flip operations to form target.
	
Example 2:

	Input: target = "101"
	Output: 3
	Explanation: "000" -> "111" -> "100" -> "101".


Example 3:

	Input: target = "00000"
	Output: 0

	
Example 4:

	Input: target = "001011101"
	Output: 5

	



Note:


	1 <= target.length <= 10^5
	target[i] is either '0' or '1'.


### 解析


根据题意，就是房间里面有一排从左到右排列的灯泡的开关，开关只有开和关两种状态，用 1 和 0 表示，这排开关的索引为 0 到 n-1 。我们可以进行一种操作，这种操作要求将索引 i 到 n-1 的灯泡都要按一次。题目给出了一个开关的排列目标 target ，求经过多少次这种操作就可以让开关形成 target 的样式。

其实这个题看起来很难，我一开始不会做，但是看完大神的解题思路发现真简单，就是一个找规律的题：

* 当开头为 1 的时候举例 target 为 10 ，变化过程为：00->11->10
* 当开头为 0 的时候举例 target 为 01 ，变化过程为：00->01
* 当开头为 1 的时候举例 target 为 100 ，变化过程为：000->111->100
* 当开头为 1 的时候举例 target 为 101 ，变化过程为：000->111->100->101
* 当开头为 0 的时候举例 target 为 010 ，变化过程为：000->011->010

很明显有规律可循：

* 当 target 的第一个字母是 0 的时候，我们将 result 初始化 0 ，第一个字母是 1 的时候，初始化为 1
* 然后从第 2 个字符开始向后遍历，判断如果当前字符不等于前一个字符则 result 加一
* 遍历结束返回 result


### 解答
				

	class Solution(object):
	    def minFlips(self, target):
	        """
	        :type target: str
	        :rtype: int
	        """
	        if target[0] == '0':
	            result = 0
	        elif target[0] == '1':
	            result = 1
	        for i in range(1, len(target)):
	            if target[i]!=target[i-1]:
	                result += 1
	        return result
            	      
			
### 运行结果

	Runtime: 84 ms, faster than 57.69% of Python online submissions for Bulb Switcher IV.
	Memory Usage: 17.5 MB, less than 13.46% of Python online submissions for Bulb Switcher IV.


原题链接：https://leetcode.com/problems/bulb-switcher-iv/



您的支持是我最大的动力
