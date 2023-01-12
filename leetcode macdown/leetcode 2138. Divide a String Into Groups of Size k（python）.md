leetcode 2138. Divide a String Into Groups of Size k （python）


由亚马逊公司赞助的 Leetcode Weekly Contest 276 ，优秀者还能获得亚马逊公司的面试机会（慕了），看了一下榜单第一名是个北大的选手，只用了 8 分钟就解完了 4 道题，佩服得六体投地【狗头】，吾辈榜样及楷模，估计我看完四道题目描述估计都得 8 分钟，惭愧。本文介绍的是周赛第一道题目，难度 Easy ，基本是个送分题目。

### 每日经典

《道德经》 ——老子（春秋）

天地不仁，以万物为刍狗；圣人不仁，以百姓为刍狗。

天地之间，其犹橐(tuó)籥(yuè)乎？虚而不屈，动而愈出。

### 描述

A string s can be partitioned into groups of size k using the following procedure:

* The first group consists of the first k characters of the string, the second group consists of the next k characters of the string, and so on. Each character can be a part of exactly one group.
* For the last group, if the string does not have k characters remaining, a character fill is used to complete the group.

Note that the partition is done so that after removing the fill character from the last group (if it exists) and concatenating all the groups in order, the resultant string should be s.

Given the string s, the size of each group k and the character fill, return a string array denoting the composition of every group s has been divided into, using the above procedure.



Example 1:

	Input: s = "abcdefghi", k = 3, fill = "x"
	Output: ["abc","def","ghi"]
	Explanation:
	The first 3 characters "abc" form the first group.
	The next 3 characters "def" form the second group.
	The last 3 characters "ghi" form the third group.
	Since all groups can be completely filled by characters from the string, we do not need to use fill.
	Thus, the groups formed are "abc", "def", and "ghi".

	
Example 2:


	Input: s = "abcdefghij", k = 3, fill = "x"
	Output: ["abc","def","ghi","jxx"]
	Explanation:
	Similar to the previous example, we are forming the first three groups "abc", "def", and "ghi".
	For the last group, we can only use the character 'j' from the string. To complete this group, we add 'x' twice.
	Thus, the 4 groups formed are "abc", "def", "ghi", and "jxx".




Note:

	1 <= s.length <= 100
	s consists of lowercase English letters only.
	1 <= k <= 100
	fill is a lowercase English letter.


### 解析

根据题意，要求将字符串 s 划分为大小为 k 的组，规则如下：

* 第一组由字符串的前 k 个字符组成，第二组由字符串的接下来的 k 个字符组成，依此类推
* 对于最后一组，如果字符串剩余的不够 k 个字符，则使用字符 fill 来填充
* 最后要保证将最后一个组删除填充字符 fill 之后，这些组的字符串按顺序连接应该还能还原是 s。

给定字符串 s、每个组 k 的大小和字符 fill ，返回一个字符串数组。

其实基本上读完一遍题目我们就能理解了，这就是个简单的字符串拼接题目，按照题意写代码即可，没什么难度。时间复杂度是 O(n) ，空间复杂度也是 O(n) 。如果连这个都做不出来那只能回家种地或者卖红薯去了。


### 解答
				

	class Solution(object):
	    def divideString(self, s, k, fill):
	        """
	        :type s: str
	        :type k: int
	        :type fill: str
	        :rtype: List[str]
	        """
	        result = []
	        s = list(s)
	        while len(s) >= k:
	            t = ""
	            for _ in range(k):
	                t += s.pop(0)
	            result.append(t)
	        if len(s) > 0:
	            result.append(''.join(s + [fill] * (k-len(s))))
	        return result
            	      
			
### 运行结果

	Runtime: 39 ms
	Memory Usage: 13.5 MB


原题链接：https://leetcode.com/contest/weekly-contest-276/problems/divide-a-string-into-groups-of-size-k/



您的支持是我最大的动力
