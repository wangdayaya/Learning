leetcode  2380. Time Needed to Rearrange a Binary String（python）




### 描述


You are given a binary string s. In one second, all occurrences of "01" are simultaneously replaced with "10". This process repeats until no occurrences of "01" exist.

Return the number of seconds needed to complete this process.


Example 1:

	Input: s = "0110101"
	Output: 4
	Explanation: 
	After one second, s becomes "1011010".
	After another second, s becomes "1101100".
	After the third second, s becomes "1110100".
	After the fourth second, s becomes "1111000".
	No occurrence of "01" exists any longer, and the process needed 4 seconds to complete,
	so we return 4.

	
Example 2:

	Input: s = "11100"
	Output: 0
	Explanation:
	No occurrence of "01" exists in s, and the processes needed 0 seconds to complete,
	so we return 0.




Note:

	1 <= s.length <= 1000
	s[i] is either '0' or '1'.


### 解析

根据题意，给定一个二进制字符串 s 。 在一秒钟内，所有出现的“01”同时替换为“10”。 重复此过程，直到不存在“01”。返回完成此过程所需的秒数。

因为数据规模很小，其实如果你懂 python 的话，里面有一个内置函数 replace 就可以实现暴力将字符串 s 中的 01 同时替换为 10 ，只需要使用 result 来进行计数操作次数即可。

时间复杂度为 O(N^2)。空间复杂度为 O(1) 。

### 解答

	class Solution(object):
	    def secondsToRemoveOccurrences(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        result = 0
	        while '01' in s:
	            s = s.replace('01','10')
	            result += 1
	        return result

### 运行结果

	
	103 / 103 test cases passed.
	Status: Accepted
	Runtime: 226 ms
	Memory Usage: 13.4 MB

### 解析

我们具体分析例子就会发现，其实将 01 替换成 10 能看作将 1 向左移动一位的操作，如果 1 前面是 0 ，则会向左移动一位进行交换，但是碰到 11 却不能同时移动，右边的 1 比左边的 1 往左走到最终的位置会比左边的 1 多一秒。举个例子就明白了，开始是 0011 ，要经过以下 3s ：

		0101 这里只能移动左边的 1
		1010 这里同时移动两个 1 ，左边的 1 已经到达最终的位置，用了 2 s
		1100 这里只能移动右边的 1 往左走，到达最终的位置，用了 3 s
		
所以我们根据上面的思路从左到右遍历 s ，不断记录 1 前面的 0 的个数，当遍历到 1 ，就更新所用的最小时间 result ，最后遍历结束直接返回 result 即可。

时间复杂度为 O(N)。空间复杂度为 O(1) 。

### 解答

	class Solution(object):
	    def secondsToRemoveOccurrences(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        result, cnt = 0, 0
	        for c in s:
	            if c == '0': 
	                cnt += 1
	            elif cnt: 
	                result = max(result + 1, cnt)
	        return result
	


### 运行结果

	103 / 103 test cases passed.
	Status: Accepted
	Runtime: 40 ms
	Memory Usage: 13.4 MB

### 原题链接


https://leetcode.com/contest/biweekly-contest-85/problems/time-needed-to-rearrange-a-binary-string/

您的支持是我最大的动力
