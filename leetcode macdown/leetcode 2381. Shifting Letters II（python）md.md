leetcode  2381. Shifting Letters II （python）




### 描述



You are given a string s of lowercase English letters and a 2D integer array shifts where shifts[i] = [start<sub>i</sub>, end<sub>i</sub>, direction<sub>i</sub>]. For every i, shift the characters in s from the index start<sub>i</sub> to the index end<sub>i</sub> (inclusive) forward if direction<sub>i</sub> = 1, or shift the characters backward if direction<sub>i</sub> = 0. Shifting a character forward means replacing it with the next letter in the alphabet (wrapping around so that 'z' becomes 'a'). Similarly, shifting a character backward means replacing it with the previous letter in the alphabet (wrapping around so that 'a' becomes 'z').

Return the final string after all such shifts to s are applied.

Example 1:

	Input: s = "abc", shifts = [[0,1,0],[1,2,1],[0,2,1]]
	Output: "ace"
	Explanation: Firstly, shift the characters from index 0 to index 1 backward. Now s = "zac".
	Secondly, shift the characters from index 1 to index 2 forward. Now s = "zbd".
	Finally, shift the characters from index 0 to index 2 forward. Now s = "ace".

	
Example 2:


	Input: s = "dztz", shifts = [[0,0,0],[1,1,1]]
	Output: "catz"
	Explanation: Firstly, shift the characters from index 0 to index 0 backward. Now s = "cztz".
	Finally, shift the characters from index 1 to index 1 forward. Now s = "catz".



Note:

* 	1 <= s.length, shifts.length <= 5 * 10^4
* 	shifts[i].length == 3
* 	0 <= start<sub>i</sub> <= end<sub>i</sub> < s.length
* 	0 <= direction<sub>i</sub> <= 1
* 	s consists of lowercase English letters.


### 解析

根据题意，给定一个小写英文字母的字符串 s 和一个 2D 整数数组 shifts  ， shifts[i] = [start<sub>i</sub>, end<sub>i</sub>, direction<sub>i</sub>]。 对于每个 i ，如果 direction<sub>i</sub> = 1，则将 s 中从索引 start<sub>i</sub> 到索引 end<sub>i</sub>（包括）的字符向前移动 ，如果 direction<sub>i</sub> = 0，则向后移动字符。 向前移动一个字符意味着用字母表中的下一个字母替换它（注意'z'下一个是'a'）。 类似地，向后移动一个字符意味着用字母表中的前一个字母替换它（注意'a'下一个是'z'）。在应用了对 s 的所有此类移位后，返回最终字符串。

这道题的本质就是考察差分算法，不懂的先进入[这里预习](https://blog.csdn.net/weixin_45629285/article/details/111146240)，差分算法的主要用于频繁对原始数组的某个区间的所有元素进行加 n 操作（n 可以是正数也可以是负数），和这道题目类似，其实我们只需要遍历 shifts ，对于第 i 个字符假如向前就 +1 ，向后就 -1 ，我们可以计算出最后 s[i] 到底是向前或者向后移动几位就可以了。

我们定义一个差分数组 sub ，sub 的前缀和构成的数组 cur 可以表示第 i 个字符最终是向前还是向后移动了几位。所以关键就在于我们怎么求出差分数组，根据差分数组的规律，我们在遍历 shifts 的时候，对于区间 [start, end] 的所有元素都加 n 的操作，可以变为将 sub[start] 加 n ，将 sub[end+1] 减 n 的操作，这样遍历 shifts 结束就可以得到完成的差分数组 sub ，然后经过计算就可以得到差分数组的前缀和 cur ，这样就可以知道每个 s[i] 的最终移动方向和距离，只需要对每个字符经过简单的字符变换和拼接变成新的 result 即可。

时间复杂度为 O(N)，空间复杂度为 O(N) 。




### 解答

	class Solution(object):
	    def shiftingLetters(self, s, shifts):
	        """
	        :type s: str
	        :type shifts: List[List[int]]
	        :rtype: str
	        """
	        N = len(s)
	        sub = [0] * (N+1)
	        for start,end,dir in shifts:
	            sub[start] += 1 if dir == 1 else -1
	            sub[end+1] -= 1 if dir == 1 else -1
	        presum = 0
	        cur = []
	        for i in range(N):
	            presum += sub[i]
	            cur.append(presum)
	        c2i = {chr(i+97):i for i in range(26)}
	        i2c = {i:chr(i+97) for i in range(26)}
	        result = ''
	        for i,c in enumerate(s):
	            result += i2c[(c2i[c] + cur[i]) % 26]
	        return result

### 运行结果

	Runtime: 1832 ms, faster than 100.00% of Python online submissions for Shifting Letters II.
	Memory Usage: 40.5 MB, less than 100.00% of Python online submissions for Shifting Letters II.

### 原题链接

https://leetcode.com/contest/biweekly-contest-85/problems/shifting-letters-ii/


您的支持是我最大的动力
