leetcode 856. Score of Parentheses （python）

### 描述


Given a balanced parentheses string s, return the score of the string.

The score of a balanced parentheses string is based on the following rule:

* "()" has score 1.
* AB has score A + B, where A and B are balanced parentheses strings.
* (A) has score 2 * A, where A is a balanced parentheses string.


Example 1:

	Input: s = "()"
	Output: 1

	
Example 2:

	
	Input: s = "(())"
	Output: 2

Example 3:


	Input: s = "()()"
	Output: 2
	
Example 4:


	Input: s = "(()(()))"
	Output: 6
	
Example 5:




Note:

	
	
	2 <= s.length <= 50
	s consists of only '(' and ')'.
	s is a balanced parentheses string.



### 解析

根据题意，就是给出来一个圆括号字符串，要求计算所得的分数，规则是：

* 对于 () 分数为 1
* 对于 A+B 型的组合，分数为 A 的分数加上 B 的分数
* 对于 (A) 型的组合，分数为 A 的分数的 2 倍

其实对于括号类型的题目，天生的特性完全可以用栈的结构来进行解答的，这道题也一样但需要结合题意进行一番变化。这里主要用例子 4 来解释，当我们遇到 ( 时候就加深括号的深度，新的深度为 0 ，遇到 ) 的时候就将之前的分数乘二，除了 () 的分数为 1 ，得到的栈的结果如下：

* [0] init stack
* [0, 0] after parsing (
* [0, 0, 0] after (
* [0, 1] after )
* [0, 1, 0] after (
* [0, 1, 0, 0] after (
* [0, 1, 1] after )
* [0, 3] after )
* [6] after )

最后得到的 stack[-1] 即为答案。

### 解答
				
	class Solution(object):
	    def scoreOfParentheses(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        stack = [0]
	        for c in s:
	            if c == '(':
	                stack.append(0)
	            elif c == ')':
	                t = stack.pop()
	                stack[-1] += max(2*t, 1)
	        return stack[-1]

            	      
			
### 运行结果

	
	Runtime: 31 ms, faster than 5.22% of Python online submissions for Score of Parentheses.
	Memory Usage: 13.5 MB, less than 59.70% of Python online submissions for Score of Parentheses.

### 解析

这种乘二的操作也很适合位移运算，其实我们只要知道每对 () 的深度 d 就可以得到这个括号对的分数为 2^d ，也就是 1>>d ，并加入到结果 result 中去。如例子 4 所示 (()(())) ：

* 其中出现的第一个出现的 () 深度为 1 ，其分数为 1>>1
* 第二个出现的 () 深度 2 ，其分数为 1>>2 
* 将两者加起来即为答案 6 

### 解答

	class Solution(object):
	    def scoreOfParentheses(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        result, balance = 0, 0
	        for i,c in enumerate(s):
	            if c=='(':
	                balance += 1
	            else:
	                balance -= 1
	                if s[i-1] == '(':
	                    result += 1 << balance
	        return result

### 运行结果

	Runtime: 19 ms, faster than 42.54% of Python online submissions for Score of Parentheses.
	Memory Usage: 13.4 MB, less than 59.70% of Python online submissions for Score of Parentheses.

原题链接：https://leetcode.com/problems/score-of-parentheses/



您的支持是我最大的动力
