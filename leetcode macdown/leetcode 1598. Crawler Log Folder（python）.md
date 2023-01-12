leetcode  1598. Crawler Log Folder（python）

### 描述

The Leetcode file system keeps a log each time some user performs a change folder operation.

The operations are described below:

* "../" : Move to the parent folder of the current folder. (If you are already in the main folder, remain in the same folder).
* "./" : Remain in the same folder.
* "x/" : Move to the child folder named x (This folder is guaranteed to always exist).

You are given a list of strings logs where logs[i] is the operation performed by the user at the ith step.

The file system starts in the main folder, then the operations in logs are performed.

Return the minimum number of operations needed to go back to the main folder after the change folder operations.





Example 1:

![](https://assets.leetcode.com/uploads/2020/09/09/sample_11_1957.png)

	Input: logs = ["d1/","d2/","../","d21/","./"]
	Output: 2
	Explanation: Use this change folder operation "../" 2 times and go back to the main folder.

	
Example 2:

![](https://assets.leetcode.com/uploads/2020/09/09/sample_22_1957.png)

	Input: logs = ["d1/","d2/","./","d3/","../","d31/"]
	Output: 3


Example 3:


	Input: logs = ["d1/","../","../","../"]
	Output: 0
	



Note:

	1 <= logs.length <= 10^3
	2 <= logs[i].length <= 10
	logs[i] contains lowercase English letters, digits, '.', and '/'.
	logs[i] follows the format described in the statement.
	Folder names consist of lowercase English letters and digits.


### 解析

根据题意，每次用户执行更改文件夹操作时，Leetcode 文件系统都会保存日志。操作说明如下：

* "../" : 移动到当前文件夹的父文件夹。 （如果已经在根文件夹中，则保持不变）
* "./" : 仍然留在当前文件夹中
* "x/" ：移动到名为 x 的子文件夹（该文件夹题目会保证存在）。

题目给出一个字符串列表 logs ，其中 logs[i] 是用户在第 i 步执行的操作。文件系统从根文件夹开始，然后执行日志中的操作。返回更改文件夹操作后返回根文件夹所需的最少操作次数。

其实这个题很简单，在我看来就是考察了栈的基本操作，思路如下：

* 初始化一个空列表 result ，用来保存经过的文件夹
* 遍历 logs 中的每个操作 log ，如果指令为 '../' ，如果 result 不为空，则弹出最后一个操作，表示回退到上一个文件夹，然后继续下一个 log 操作 ；如果指令为  './' ，什么都不做直接继续下一个 log 操作；如果是其他情况直接加入到 result ，表示经过了该文件夹
* 遍历结束查看 result 的长度即为返回根文件夹的最少操作次数


### 解答
				
	class Solution(object):
	    def minOperations(self, logs):
	        """
	        :type logs: List[str]
	        :rtype: int
	        """
	        result = []
	        for log in logs:
	            if '../' == log:
	                if result:
	                    result.pop(-1)
	                continue
	            if './' == log:
	                continue
	            result.append(log)
	        return len(result)
	            
	                
	           
            	      
			
### 运行结果

	Runtime: 36 ms, faster than 48.94% of Python online submissions for Crawler Log Folder.
	Memory Usage: 13.8 MB, less than 38.30% of Python online submissions for Crawler Log Folder.


原题链接：https://leetcode.com/problems/crawler-log-folder/



您的支持是我最大的动力
