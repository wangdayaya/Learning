leetcode  2325. Decode the Message（python）




### 描述

You are given the strings key and message, which represent a cipher key and a secret message, respectively. The steps to decode message are as follows:

* Use the first appearance of all 26 lowercase English letters in key as the order of the substitution table.
* Align the substitution table with the regular English alphabet.
* Each letter in message is then substituted using the table.
* Spaces ' ' are transformed to themselves.
* For example, given key = "happy boy" (actual key would have at least one instance of each letter in the alphabet), we have the partial substitution table of ('h' -> 'a', 'a' -> 'b', 'p' -> 'c', 'y' -> 'd', 'b' -> 'e', 'o' -> 'f').
Return the decoded message.



Example 1:

![](https://assets.leetcode.com/uploads/2022/05/08/ex1new4.jpg)

	Input: key = "the quick brown fox jumps over the lazy dog", message = "vkbs bs t suepuv"
	Output: "this is a secret"
	Explanation: The diagram above shows the substitution table.
	It is obtained by taking the first appearance of each letter in "the quick brown fox jumps over the lazy dog".


	
Example 2:


![](https://assets.leetcode.com/uploads/2022/05/08/ex2new.jpg)

	Input: key = "eljuxhpwnyrdgtqkviszcfmabo", message = "zwx hnfx lqantp mnoeius ycgk vcnjrdb"
	Output: "the five boxing wizards jump quickly"
	Explanation: The diagram above shows the substitution table.
	It is obtained by taking the first appearance of each letter in "eljuxhpwnyrdgtqkviszcfmabo".




Note:

	26 <= key.length <= 2000
	key consists of lowercase English letters and ' '.
	key contains every letter in the English alphabet ('a' to 'z') at least once.
	1 <= message.length <= 2000
	message consists of lowercase English letters and ' '.


### 解析

根据题意，给定字符串 key 和 message，它们分别代表密码密钥和秘密消息，要求返回解码的秘密消息。 消息解码步骤如下：

* 使用 key 中所有第一次出现的 26 个小写英文字母作为替换表的顺序
* 将替换表与常规英文字母对齐
* 然后使用该表替换消息中的每个字母
* 空格被转换为空格即可

例如，给定 key = "happy boy"，我们可以得到  ('h' -> 'a', 'a' -> 'b', 'p' -> 'c', 'y' -> 'd', 'b' -> 'e', 'o' -> 'f'）。



这道题其实就是考察一个字符之间的映射，我们先将 key 中第一次出现的不同的字符收集起来放入 seen 中，然后从左到右对应 a-z 的映射关系放入 d 中，使用 d 解码 message 得到的结果返回即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答
				

	class Solution(object):
	    def decodeMessage(self, key, message):
	        """
	        :type key: str
	        :type message: str
	        :rtype: str
	        """
	        key  = key.replace(' ', '')
	        d = {}
	        d[' '] = ' '
	        seen = []
	        for c in key:
	            if c not in seen:
	                seen.append(c)
	        for i,c in enumerate(seen):
	            if i==26:
	                break
	            d[c] = chr(ord('a')+i)
	        result = ''
	        for m in list(message):
	            result += d[m]
	        return result

            	      
			
### 运行结果


	
	68 / 68 test cases passed.
	Status: Accepted
	Runtime: 18 ms
	Memory Usage: 13.7 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-300/problems/decode-the-message/


您的支持是我最大的动力
