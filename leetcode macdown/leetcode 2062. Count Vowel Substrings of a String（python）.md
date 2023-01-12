leetcode  2062. Count Vowel Substrings of a String（python）

### 描述

A substring is a contiguous (non-empty) sequence of characters within a string.

A vowel substring is a substring that only consists of vowels ('a', 'e', 'i', 'o', and 'u') and has all five vowels present in it.

Given a string word, return the number of vowel substrings in word.

 



Example 1:


	Input: word = "aeiouu"
	Output: 2
	Explanation: The vowel substrings of word are as follows (underlined):
	- "aeiouu"
	- "aeiouu"
	
Example 2:

	Input: word = "unicornarihan"
	Output: 0
	Explanation: Not all 5 vowels are present, so there are no vowel substrings.


Example 3:

	Input: word = "cuaieuouac"
	Output: 7
	Explanation: The vowel substrings of word are as follows (underlined):
	- "cuaieuouac"
	- "cuaieuouac"
	- "cuaieuouac"
	- "cuaieuouac"
	- "cuaieuouac"
	- "cuaieuouac"
	- "cuaieuouac"

	
Example 4:

	Input: word = "bbaeixoubb"
	Output: 0
	Explanation: The only substrings that contain all five vowels also contain consonants, so there are no vowel substrings.



Note:
	
	1 <= word.length <= 100
	word consists of lowercase English letters only.

### 解析

根据题意，给出了一个字符串单词 word ，让我们返回在 word 中有多少个 vowel substrings ，题目中也给出了 vowel substrings 的定义，那就是就是非空子字符串且恰好包含了 a、e、i、o、u 五种小写字母，但是个数不限。

先使用暴力解法，其实很简单：

* 初始化结果 reuslt 为 0
* 遍历 range(5, len(word) + 1) 中的每个元素 L ，表示当前的子字符串的长度
* 然后遍历 ange(len(word) - L + 1) 中的每个元素的 i ，表示从索引 i 开始找子字符串
* 通过内置函数 collections.Counter 计算 word[i:i + L] 中的字符，如果符合 vowel substrings 的条件，就将结果 result 加一
* 遍历结束返回 result 即可

从运行结果来看，虽然是超过 100% ，但是不代表暴力解法有效，可能是新题，提交答案的人数太少了。人要对自己的能力有清醒的认识。
### 解答
					
	class Solution(object):
	    def countVowelSubstrings(self, word):
	        """
	        :type word: str
	        :rtype: int
	        """
	        result = 0
	        for L in range(5, len(word) + 1):
	            for i in range(len(word) - L + 1):
	                c = collections.Counter(word[i:i + L])
	                keys = c.keys()
	                if 'a' in keys and 'e' in keys and 'i' in keys and 'o' in keys and 'u' in keys and len(keys) == 5:
	                    result += 1
	        return result

            	      
			
### 运行结果

	Runtime: 2676 ms, faster than 100.00% of Python online submissions for Count Vowel Substrings of a String.
	Memory Usage: 13.4 MB, less than 100.00% of Python online submissions for Count Vowel Substrings of a String.

### 解析

其实可以用集合函数 set 对上面的代码进行精简。


### 解答

	class Solution(object):
	    def countVowelSubstrings(self, word):
	        """
	        :type word: str
	        :rtype: int
	        """
	        return sum(set(word[i:i+L]) == set('aeiou') for L in range(5, len(word)+1) for i in range(len(word)-L+1))
	        
### 运行结果

	Runtime: 316 ms, faster than 100.00% of Python online submissions for Count Vowel Substrings of a String.
	Memory Usage: 13.5 MB, less than 100.00% of Python online submissions for Count Vowel Substrings of a String.

### 解析

可以使用类似滑动窗口的方法解题，思路很简单：

* 先初始化结果 result 为 0 ，原音集合 vowels 为 {'a','e','i','o','u'} 
* 因为 vowel substring 最短为 5 ，所以遍历 range(len(word)-4) 中的每个元素 i ，表示子字符串的开始索引
* 如果 word[i] 在 vowel 中，则新建一个集合 s 将  word[i] 加入进去
* 然后遍历 range(i+1, len(word)) 中的每个元素 j ，表示遍历 i 索引之后的字符，如果 word[j] 不在 vowels 说明 word[i:j] 不能构成 vowel substring ，直接 break 进行之后的循环，否则将 word[j]  加入 s 中，如果 s 的长度为 5 说明可以构成 vowel substring 了，result 加一
* 上述两层循环结束返回 result

从运行结果可以看出，运算时间大幅度提升。

### 解答

	class Solution(object):
	    def countVowelSubstrings(self, word):
	        """
	        :type word: str
	        :rtype: int
	        """
	        result = 0
	        vowels = {'a','e','i','o','u'}
	        for i in range(len(word)-4):
	            if word[i] in vowels:
	                s = set(word[i])
	                for j in range(i+1, len(word)):
	                    if word[j] not in vowels:
	                        break
	                    s.add(word[j])
	                    if len(s) == 5:
	                        result += 1
	        return result
	        
### 运行结果

	Runtime: 52 ms, faster than 100.00% of Python online submissions for Count Vowel Substrings of a String.
	Memory Usage: 13.5 MB, less than 100.00% of Python online submissions for Count Vowel Substrings of a String.


原题链接：https://leetcode.com/problems/count-vowel-substrings-of-a-string/



您的支持是我最大的动力
