leetcode  1974. Minimum Time to Type Word Using Special Typewriter（python）

### 描述

There is a special typewriter with lowercase English letters 'a' to 'z' arranged in a circle with a pointer. A character can only be typed if the pointer is pointing to that character. The pointer is initially pointing to the character 'a'.

![](https://assets.leetcode.com/uploads/2021/07/31/chart.jpg)

Each second, you may perform one of the following operations:

* Move the pointer one character counterclockwise or clockwise.
* Type the character the pointer is currently on.

Given a string word, return the minimum number of seconds to type out the characters in word.

Example 1:


	Input: word = "abc"
	Output: 5
	Explanation: 
	The characters are printed as follows:
	- Type the character 'a' in 1 second since the pointer is initially on 'a'.
	- Move the pointer clockwise to 'b' in 1 second.
	- Type the character 'b' in 1 second.
	- Move the pointer clockwise to 'c' in 1 second.
	- Type the character 'c' in 1 second.
	
Example 2:


	Input: word = "bza"
	Output: 7
	Explanation:
	The characters are printed as follows:
	- Move the pointer clockwise to 'b' in 1 second.
	- Type the character 'b' in 1 second.
	- Move the pointer counterclockwise to 'z' in 2 seconds.
	- Type the character 'z' in 1 second.
	- Move the pointer clockwise to 'a' in 1 second.
	- Type the character 'a' in 1 second.

Example 3:


	Input: word = "zjpc"
	Output: 34
	Explanation:
	The characters are printed as follows:
	- Move the pointer counterclockwise to 'z' in 1 second.
	- Type the character 'z' in 1 second.
	- Move the pointer clockwise to 'j' in 10 seconds.
	- Type the character 'j' in 1 second.
	- Move the pointer clockwise to 'p' in 6 seconds.
	- Type the character 'p' in 1 second.
	- Move the pointer counterclockwise to 'c' in 13 seconds.
	- Type the character 'c' in 1 second.
	



Note:

	
	1 <= word.length <= 100
	word consists of lowercase English letters.

### 解析


根据题意， 给出了一个用 26 个小写英文字母按顺序组成的一个圆盘，当指针指向某个字母的时候，需要打印这个字母，最开始指针指向的是字母 a 。

圆盘可以顺时针或者逆时针转动，我们可以每次进行两种操作：

* 顺时针或者逆时针转动圆盘，让指针指向 word 中的某个字符，经过一个字符就消耗一秒，移动几次就是需要几秒
* 打印这个字符，只需要 1 秒

题目中给出了一个单词 word ，让我们计算打印这个 word 需要的最少的时间秒数。

其实这个题很简单，就是遍历 word 中的每个字符，然后找出从上一个字符到下一个字符的指针最少移动次数，可以顺时针找也可以逆时针找，然后将所有的指针移动时间和字母的打印时间都加起来就可以了。关键在于顺时针和逆时针的时间计算。

### 解答
				


	class Solution(object):
	    def minTimeToType(self, word):
	        """
	        :type word: str
	        :rtype: int
	        """
	        result = 0
	        for i, letter in enumerate(word):
	            if i == 0:
	                result += 1 + min(ord(letter)-ord('a'), 122-ord(letter)+1)
	            else:
	                # 122 是 z 的 ascii 码 ，97 是 a 的 ascii 码
	                if ord(letter) > ord(word[i-1]):
	                    result += 1 + min(ord(letter)-ord(word[i-1]), 122-ord(letter)+1+ord(word[i-1])-97)
	                elif ord(letter) < ord(word[i-1]):
	                    result += 1 + min(ord(word[i-1])-ord(letter), 122-ord(word[i-1])+1+ord(letter)-97)
	                else:
	                    result += 1
	        return result       	      
			
### 运行结果

	Runtime: 41 ms, faster than 8.36% of Python online submissions for Minimum Time to Type Word Using Special Typewriter.
	Memory Usage: 13.4 MB, less than 53.99% of Python online submissions for Minimum Time to Type Word Using Special Typewriter.


### 解析


其实更简单的方法，使用贪心算法求解即可。

### 解答


	class Solution(object):
	    def minTimeToType(self, word):
	        """
	        :type word: str
	        :rtype: int
	        """
	        s = "abcdefghojklmnopqrstuvwxyz"
	        result = 0
	        cur = "a"
	        for letter in word:
	            if cur == letter:
	                result += 1
	                continue
	            result += min(abs(ord(letter)-ord(cur)),26-abs(ord(letter)-ord(cur))) + 1
	            cur = letter
	        return result

### 运行结果

	Runtime: 20 ms, faster than 73.38% of Python online submissions for Minimum Time to Type Word Using Special Typewriter.
	Memory Usage: 13.5 MB, less than 53.99% of Python online submissions for Minimum Time to Type Word Using Special Typewriter.




原题链接：https://leetcode.com/problems/minimum-time-to-type-word-using-special-typewriter/



您的支持是我最大的动力
