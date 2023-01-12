

### 描述


Given a 0-indexed string word and a character ch, reverse the segment of word that starts at index 0 and ends at the index of the first occurrence of ch (inclusive). If the character ch does not exist in word, do nothing.

* For example, if word = "abcdefd" and ch = "d", then you should reverse the segment that starts at 0 and ends at 3 (inclusive). The resulting string will be "dcbaefd".

Return the resulting string.




Example 1:

	Input: word = "abcdefd", ch = "d"
	Output: "dcbaefd"
	Explanation: The first occurrence of "d" is at index 3. 
	Reverse the part of word from 0 to 3 (inclusive), the resulting string is "dcbaefd".
	
Example 2:

	Input: word = "xyxzxe", ch = "z"
	Output: "zxyxxe"
	Explanation: The first and only occurrence of "z" is at index 3.
	Reverse the part of word from 0 to 3 (inclusive), the resulting string is "zxyxxe".

Example 3:

	Input: word = "abcd", ch = "z"
	Output: "abcd"
	Explanation: "z" does not exist in word.
	You should not do any reverse operation, the resulting string is "abcd".

Note:

	1 <= word.length <= 250
	word consists of lowercase English letters.
	ch is a lowercase English letter.

### 解析

根据题意，就是给出了一个英文字符串 word ，又给出了一个字符 ch ，如果当 ch 存在于 word 中，那就让 word 从头开始到这个 ch 位置的子字符串进行反转，最后返回处理后的字符串结果，如果 ch 没有存在于 word 中，那么不做任何的操作。思路很简单：

* 如果 ch 不在 word 中，直接返回 word
* 否则找出 ch 在 word 中的索引 i ，直接返回 word[:i+1][::-1] + word[i+1:] 即可

### 解答
				
	class Solution(object):
	    def reversePrefix(self, word, ch):
	        """
	        :type word: str
	        :type ch: str
	        :rtype: str
	        """
	        if ch not in word:
	            return word
	        i = word.index(ch)
	        return word[:i+1][::-1] + word[i+1:]
            	      
			
### 运行结果
	Runtime: 29 ms, faster than 29.96% of Python online submissions for Reverse Prefix of Word.
	Memory Usage: 13.4 MB, less than 62.55% of Python online submissions for Reverse Prefix of Word.




### 解析

其实上面的方法就是用到了内置函数，直接遍历也是可以的，直接遍历 word ，每次将字符添加到 result 的开头，一直等找到 ch 为止，然后将当前的 ch 也加到 result 的最前头，最后将剩下的部分追加到 result 结果后面就可以了。这个方法有点凑数的嫌疑，反正题很简单，代码随便浪就行了。

### 解答

	class Solution(object):
	    def reversePrefix(self, word, ch):
	        """
	        :type word: str
	        :type ch: str
	        :rtype: str
	        """
	        if ch not in word:
	            return word
	        result = ''
	        i = 0
	        for c in word:
	            if c != ch:
	                result = c + result
	                i += 1
	            else:
	                break
	        result = word[i] + result
	        return result + word[i+1:]

### 运行结果

	Runtime: 31 ms, faster than 21.59% of Python online submissions for Reverse Prefix of Word.
	Memory Usage: 13.3 MB, less than 96.92% of Python online submissions for Reverse Prefix of Word.

原题链接：https://leetcode.com/problems/reverse-prefix-of-word/


您的支持是我最大的动力
