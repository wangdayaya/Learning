leetcode  2337. Move Pieces to Obtain a String（python）




### 描述


You are given two strings start and target, both of length n. Each string consists only of the characters 'L', 'R', and '_' where:

The characters 'L' and 'R' represent pieces, where a piece 'L' can move to the left only if there is a blank space directly to its left, and a piece 'R' can move to the right only if there is a blank space directly to its right.
The character '_' represents a blank space that can be occupied by any of the 'L' or 'R' pieces.
Return true if it is possible to obtain the string target by moving the pieces of the string start any number of times. Otherwise, return false.


Example 1:

	Input: start = "_L__R__R_", target = "L______RR"
	Output: true
	Explanation: We can obtain the string target from start by doing the following moves:
	- Move the first piece one step to the left, start becomes equal to "L___R__R_".
	- Move the last piece one step to the right, start becomes equal to "L___R___R".
	- Move the second piece three steps to the right, start becomes equal to "L______RR".
	Since it is possible to get the string target from start, we return true.

	
Example 2:


	Input: start = "R_L_", target = "__LR"
	Output: false
	Explanation: The 'R' piece in the string start can move one step to the right to obtain "_RL_".
	After that, no pieces can move anymore, so it is impossible to obtain the string target from start.

Example 3:

	Input: start = "_R", target = "R_"
	Output: false
	Explanation: The piece in the string start can move only to the right, so it is impossible to obtain the string target from start.



Note:

	n == start.length == target.length
	1 <= n <= 10^5
	start and target consist of the characters 'L', 'R', and '_'.


### 解析

根据题意，给定两个字符串 start 和 target，长度均为 n 。 每个字符串仅包含字符“L”、“R”和“_”，其中：

* 字符“L”和“R”代表棋子，其中棋子“L”只有在其左侧直接有空格时才能向左移动，而“R”只有在其右侧在有空格时才能向右移动。
* 字符“_”表示可以被任何“L”或“R”块占据的空白空间。

如果可以通过移动字符串 start 任意次数来获取字符串 target ，则返回 true 。 否则，返回 false 。

由于 L 和 R 在规则的限制下不能超过对方进行移动，所以他们的相对位置应该是一样的，也就是一一对应的，我们可以将空格都去掉比较剩下的部分，如果不相同直接返回 False 。然后我们再使用双指针的方法来进行遍历，i 指向 start ，j 指向 target ，我们发现要想满足题意，不仅要保证 L 和 R 在 start 和 target 中一一对应，还要保证以下条件：

* 如果 i 指向的字符为 L ，要保证 i >= j ，这样 L 才能向左移动，不满足就直接返回 False 
* 如果 i 指向的字符为 R ，要保证 i <= j，这样 R 才能向右移动，不满足就直接返回 False 

遍历结束之后直接返回 True 即可。时间复杂度为 O(N) ，空间复杂度为 O(1) 。

### 解答

	class Solution(object):
	    def canChange(self, start, target):
	        """
	        :type start: str
	        :type target: str
	        :rtype: bool
	        """
	        if start.replace('_', '') != target.replace("_", ""):
	            return False
	        j = 0
	        for i, c in enumerate(start):
	            if c == '_':    
	                continue
	            while target[j] == '_':
	                j += 1
	            if c == 'L':
	                if i < j:
	                    return False
	            if c == 'R':
	                if i > j:
	                    return False
	            j += 1
	        return True
	


### 运行结果


	125 / 125 test cases passed.
	Status: Accepted
	Runtime: 266 ms
	Memory Usage: 15.7 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-301/problems/move-pieces-to-obtain-a-string/


您的支持是我最大的动力
