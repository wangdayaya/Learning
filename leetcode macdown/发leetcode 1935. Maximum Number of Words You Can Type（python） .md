leetcode  1935. Maximum Number of Words You Can Type（python）

### 描述


There is a malfunctioning keyboard where some letter keys do not work. All other keys on the keyboard work properly.

Given a string text of words separated by a single space (no leading or trailing spaces) and a string brokenLetters of all distinct letter keys that are broken, return the number of words in text you can fully type using this keyboard.


Example 1:

	Input: text = "hello world", brokenLetters = "ad"
	Output: 1
	Explanation: We cannot type "world" because the 'd' key is broken.

	
Example 2:

	Input: text = "leet code", brokenLetters = "lt"
	Output: 1
	Explanation: We cannot type "leet" because the 'l' and 't' keys are broken.


Example 3:

	Input: text = "leet code", brokenLetters = "e"
	Output: 0
	Explanation: We cannot type either word because the 'e' key is broken.





Note:

* 1 <= text.length <= 10^4
* 0 <= brokenLetters.length <= 26
* text consists of words separated by a single space without any leading or trailing spaces.
* Each word only consists of lowercase English letters.
* brokenLetters consists of distinct lowercase English letters.


### 解析


根据题意，就是给出了一个字符串 text ，里面的单词都用一个空格分割，且前后没有空格缀，又给出了键盘上坏了的键 brokenLetters ，如果该单词恰好需要坏了的键则表示不能成功打出，求能顺利打出完整的单词的个数。思路比较简单：

* 将 text 按照空格分割成单词列表 words
* 初始化 result 为 words 的长度
* 外层循环遍历 words ，内层遍历每个单词的每个字符 c ，如果 c 存在于 brokenLetters 则 result 减一，然后直接进行下一个单词的判断
* 遍历结束得到的 result 即为结果

### 解答
				
	class Solution(object):
	    def canBeTypedWords(self, text, brokenLetters):
	        """
	        :type text: str
	        :type brokenLetters: str
	        :rtype: int
	        """
	        words = text.split(' ')
	        result = len(words)
	        for word in words:
	            for c in word:
	                if c in brokenLetters:
	                    result -= 1
	                    break
	        return result

            	      
			
### 运行结果

	
	Runtime: 20 ms, faster than 75.98% of Python online submissions for Maximum Number of Words You Can Type.
	Memory Usage: 13.5 MB, less than 96.65% of Python online submissions for Maximum Number of Words You Can Type.


### 解析


另外，可以通过内置函数完成：

* 将字符串 brokenLetters 转换成集合 b
* 将 text 按照空格分割成单词列表 words
* 遍历每个单词，如果单词与 b 的交集为 0 ，则 result 加一
* 遍历结束 result 即为结果


### 解答

	class Solution(object):
	    def canBeTypedWords(self, text, brokenLetters):
	        """
	        :type text: str
	        :type brokenLetters: str
	        :rtype: int
	        """
	        b = set(brokenLetters)
	        words = text.split()
	        result = 0
	        for t in words:
	            if len(b.intersection(t)) == 0 :
	                result += 1
	        return result

### 运行结果

	Runtime: 20 ms, faster than 75.98% of Python online submissions for Maximum Number of Words You Can Type.
	Memory Usage: 13.7 MB, less than 80.63% of Python online submissions for Maximum Number of Words You Can Type.
	
原题链接：https://leetcode.com/problems/maximum-number-of-words-you-can-type/



您的支持是我最大的动力
