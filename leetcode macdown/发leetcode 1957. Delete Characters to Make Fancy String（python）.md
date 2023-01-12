leetcode 1957. Delete Characters to Make Fancy String （python）

### 描述


A fancy string is a string where no three consecutive characters are equal.

Given a string s, delete the minimum possible number of characters from s to make it fancy.

Return the final string after the deletion. It can be shown that the answer will always be unique.




Example 1:

	
	Input: s = "leeetcode"
	Output: "leetcode"
	Explanation:
	Remove an 'e' from the first group of 'e's to create "leetcode".
	No three consecutive characters are equal, so return "leetcode".
	
Example 2:
	
	Input: s = "aaabaaaa"
	Output: "aabaa"
	Explanation:
	Remove an 'a' from the first group of 'a's to create "aabaaaa".
	Remove two 'a's from the second group of 'a's to create "aabaa".
	No three consecutive characters are equal, so return "aabaa".


Example 3:

	
	Input: s = "aab"
	Output: "aab"
	Explanation: No three consecutive characters are equal, so return "aab".
	




Note:


	1 <= s.length <= 10^5
	s consists only of lowercase English letters.

### 解析


根据题意，就是给出一个字符串 s ，然后经过删除最少个字符，将 s 变成一个 fancy 字符串。题目中定义了 fancy 字符串就是没有包含三个连续相同字符的字符串。

理解了题意，思路就出来了：

* 如果 s 长度小于 3 则肯定是 fancy 字符串
* 初始化结果 r 为 s[0] ，计数器 count 为 1 ，表示当前字符连续出现的次数
* 从第二个字符开始，遍历 s 中的字符，如果当前字符 c 等于前一个字符，count 加一，如果 count<3 ，将 c 拼接到 r 之后，否则不满足 fancy 条件，直接进入下一个字符的判断
* 如果当前字符不等于前一个字符，表示是新字符，可以直接将 c 拼接到 r 后面，计数器重置为 1 
* 遍历结束，得到的 r 即为结果

测试结果发现尽管能通过，但是速度太慢了。
### 解答
				

	class Solution(object):
	    def makeFancyString(self, s):
	        """
	        :type s: str
	        :rtype: str
	        """
	        if len(s)<=2:return s
	        r = s[0]
	        count = 1
	        for i in range(1, len(s)):
	            c = s[i]
	            if c==s[i-1]:
	                count += 1
	                if count<3:
	                    r += c
	                else:
	                    continue
	            else:
	                count = 1
	                r += c
	        return r
            	      
			
### 运行结果


	Runtime: 8680 ms, faster than 15.24% of Python online submissions for Delete Characters to Make Fancy String.
	Memory Usage: 19.2 MB, less than 41.90% of Python online submissions for Delete Characters to Make Fancy String.


### 解析

上面的解法有点太啰嗦了，其实换个角度代码可以简洁很多：

* 如果 s 长度小于 3 则肯定是 fancy 字符串
*  s[0] 和 s[1] 都拼接到结果字符串 r 后面
*  从第三个字符开始遍历 s ，当前字符 c != r[-1] 或者 c != r[-2] ，表示当前字符与这两个字符拼接不违反 fancy 规则，可以拼接在 r 后面
*  遍历结束得到的 r 即为结果

测试结果还是很慢

### 解答
				
	class Solution(object):
	    def makeFancyString(self, s):
	        """
	        :type s: str
	        :rtype: str
	        """
	        if len(s)<=2:return s
	        r = s[0]
	        r += s[1]
	        for c in s[2:]:
	            if c!=r[-1] or c!=r[-2]:
	                r += c
	        return r
			
### 运行结果
	Runtime: 8572 ms, faster than 20.95% of Python online submissions for Delete Characters to Make Fancy String.
	Memory Usage: 16 MB, less than 90.48% of Python online submissions for Delete Characters to Make Fancy String.

原题链接：https://leetcode.com/problems/delete-characters-to-make-fancy-string/



您的支持是我最大的动力
