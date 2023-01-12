小知识，大挑战！本文正在参与“[程序员必备小知识](https://juejin.cn/post/7008476801634680869 "https://juejin.cn/post/7008476801634680869")”创作活动

### 描述


You are given a string s. We want to partition the string into as many parts as possible so that each letter appears in at most one part.

Return a list of integers representing the size of these parts.




Example 1:

	Input: s = "ababcbacadefegdehijhklij"
	Output: [9,7,8]
	Explanation:
	The partition is "ababcbaca", "defegde", "hijhklij".
	This is a partition so that each letter appears in at most one part.
	A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits s into less parts.
	
Example 2:
	
	Input: s = "eccbbbbdec"
	Output: [10]





Note:

	1 <= s.length <= 500
	s consists of lowercase English letters.


### 解析

根据题意，就是给出了一个字符串 s ，要求将 s 分成不同的区，尽可能让每种字符都只能出现在一个分区内，最后返回一个列表，包含每个分区的长度。这里的思路比较巧妙：

* 使用字典 d 来记录不同字符在字符串 s 中的最后出现的位置
* 然后初始化 j 和 anchor 都为 0 
* 遍历 s 的过程中每个字符和索引为 c 和 i ，j 是表示已经出现的字符的最远索引，如果 i == j ，那么就说明在这个位置之前出现的所有的字符种类都只出现于当前分区内，此时可以计算分区的长度并添加到 result 中，并更新 anchor 为新的分区的初始位置


### 解答
				

	class Solution(object):
	    def partitionLabels(self, s):
	        """
	        :type s: str
	        :rtype: List[int]
	        """
	        d = {c:i for i,c in enumerate(s)}
	        j = anchor = 0
	        result = []
	        for i,c in enumerate(s):
	            j = max(j, d[c])
	            if i==j:
	                result.append(i-anchor+1)
	                anchor = i+1
	        return result          	      
			
### 运行结果

	Runtime: 28 ms, faster than 69.23% of Python online submissions for Partition Labels.
	Memory Usage: 13.2 MB, less than 98.21% of Python online submissions for Partition Labels.



原题链接：https://leetcode.com/problems/partition-labels/


您的支持是我最大的动力
