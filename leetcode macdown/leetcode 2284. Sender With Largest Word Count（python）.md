leetcode  2284. Sender With Largest Word Count（python）

这是 Biweekly Contest 79 的第二题，难度 Medium ，主要考察的是计数器的操作和对双特征的排序操作。



### 描述

You have a chat log of n messages. You are given two string arrays messages and senders where messages[i] is a message sent by senders[i].

A message is list of words that are separated by a single space with no leading or trailing spaces. The word count of a sender is the total number of words sent by the sender. Note that a sender may send more than one message.

Return the sender with the largest word count. If there is more than one sender with the largest word count, return the one with the lexicographically largest name.

Note:

* Uppercase letters come before lowercase letters in lexicographical order.
* "Alice" and "alice" are distinct.



Example 1:

	Input: messages = ["Hello userTwooo","Hi userThree","Wonderful day Alice","Nice day userThree"], senders = ["Alice","userTwo","userThree","Alice"]
	Output: "Alice"
	Explanation: Alice sends a total of 2 + 3 = 5 words.
	userTwo sends a total of 2 words.
	userThree sends a total of 3 words.
	Since Alice has the largest word count, we return "Alice".

	
Example 2:

	Input: messages = ["How is leetcode for everyone","Leetcode is useful for practice"], senders = ["Bob","Charlie"]
	Output: "Charlie"
	Explanation: Bob sends a total of 5 words.
	Charlie sends a total of 5 words.
	Since there is a tie for the largest word count, we return the sender with the lexicographically larger name, Charlie.





Note:

	n == messages.length == senders.length
	1 <= n <= 10^4
	1 <= messages[i].length <= 100
	1 <= senders[i].length <= 10
	messages[i] consists of uppercase and lowercase English letters and ' '.
	All the words in messages[i] are separated by a single space.
	messages[i] does not have leading or trailing spaces.
	senders[i] consists of uppercase and lowercase English letters only.


### 解析

根据题意，有一个包含 n 条消息的聊天记录。 给定两个字符串数组 messages 和 senders，其中messages[i] 是 senders[i] 发送的消息。

消息是由单个空格分隔的单词列表，没有前导或尾随空格。 返回发送单词数最多的发件人。 如果有多个发件人字数最多，则返回按字典顺序排列的最大姓名的发件人。

读完这道题思路很明显，我们对给出的 messages 进行单词拆分，并按照 senders 中的人对发送的单词数进行统计，最后对统计出的结果按照题目给出的要求排序，首先是第一个按照第一个特征排序，看谁发送的单词数最多就返回谁，否则如果有相同多的单词数，按照第二个特征看谁的名字按照字典顺序是较大的就返回谁。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答
				
	class Solution(object):
	    def largestWordCount(self, messages, senders):
	        """
	        :type messages: List[str]
	        :type senders: List[str]
	        :rtype: str
	        """
	        d = collections.defaultdict(int)
	        for i,m in enumerate(messages):
	            sender = senders[i]
	            d[sender] += len(m.split(' '))
	        d = sorted(d.items(),key=lambda x: (x[1], x[0]), reverse=True)
	        return d[0][0]

            	      
			
### 运行结果

	65 / 65 test cases passed.
	Status: Accepted
	Runtime: 812 ms
	Memory Usage: 26.7 MB


### 原题链接

https://leetcode.com/contest/biweekly-contest-79/problems/sender-with-largest-word-count/



您的支持是我最大的动力
