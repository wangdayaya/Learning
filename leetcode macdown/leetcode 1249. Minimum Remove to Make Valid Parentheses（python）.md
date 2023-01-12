leetcode  1249. Minimum Remove to Make Valid Parentheses（python）




### 描述

Given a string s of '(' , ')' and lowercase English characters.

Your task is to remove the minimum number of parentheses ( '(' or ')', in any positions ) so that the resulting parentheses string is valid and return any valid string.

Formally, a parentheses string is valid if and only if:

* It is the empty string, contains only lowercase characters, or
* It can be written as AB (A concatenated with B), where A and B are valid strings, or
* It can be written as (A), where A is a valid string.
 



Example 1:


	Input: s = "lee(t(c)o)de)"
	Output: "lee(t(c)o)de"
	Explanation: "lee(t(co)de)" , "lee(t(c)ode)" would also be accepted
	





Note:

	1 <= s.length <= 10^5
	s[i] is either'(' , ')', or lowercase English letter.


### 解析


根据题意，给定一个由 '(' 、 ')' 和小写英文字符组成的字符串 s 。我们的任务是在任意位置删除最小数量的括号 '(' 或 ')' ，以便使的生成的带括号的字符串有效并返回。

带括号的字符串有效的情况如下：

* 没有括号，仅包含小写字符
* 它可以写成 AB（A 与 B 连接），其中 A 和 B 是有效字符串
* 它可以写成 (A)，其中 A 是一个有效的字符串 

其实这道题的题意出的挺乱，题目说的弯弯绕，但是结合几个给出的例子我们可以清楚的理解题意，其实就是 s 中的括号不正常，让我们把不正常的括号去掉，变成正常的带括号的字符串，这类题一般很自然地就想到了栈数据结构。

我们用 stack 来存放不合法括号的位置，遍历 s 中所有的字符，如果索引为 i 的位置的字符为 '(' 则将其加入 stack ；如果为 ')' 那么就要判断如果 stack 不为空，说明可以形成一个有效的括号对，则将 stack 最后一个元素弹出，如果 stack 为空说明当前括号不合法则将 s 的第 i 个索引的值变为空字符串。

遍历结束之后 stack 中留下的只有不合法的括号的索引，在 s 中将这些索引的字符都变成空字符，最后将变换之后 s 返回即可。

整个算法的时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答
				
	class Solution(object):
	    def minRemoveToMakeValid(self, s):
	        result = list(s)
	        stack = []
	        for i,c in enumerate(result):
	            if c == '(':
	                stack.append(i)
	            elif c == ')':
	                if stack:
	                    stack.pop()
	                else:
	                    result[i] = ''
	        for i in stack:
	            result[i] = ''
	        return ''.join(result)

            	      
			
### 运行结果


	Runtime: 168 ms, faster than 60.45% of Python online submissions for Minimum Remove to Make Valid Parentheses.
	Memory Usage: 15.7 MB, less than 75.08% of Python online submissions for Minimum Remove to Make Valid Parentheses.

### 原题链接

https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/

您的支持是我最大的动力
