leetcode 1915. Number of Wonderful Substrings （python）

### 描述

A wonderful string is a string where at most one letter appears an odd number of times.

* For example, "ccjjc" and "abab" are wonderful, but "ab" is not.

Given a string word that consists of the first ten lowercase English letters ('a' through 'j'), return the number of wonderful non-empty substrings in word. If the same substring appears multiple times in word, then count each occurrence separately.

A substring is a contiguous sequence of characters in a string.

 



Example 1:


	Input: word = "aba"
	Output: 4
	Explanation: The four wonderful substrings are underlined below:
	- "aba" -> "a"
	- "aba" -> "b"
	- "aba" -> "a"
	- "aba" -> "aba"
	
Example 2:

	Input: word = "aabb"
	Output: 9
	Explanation: The nine wonderful substrings are underlined below:
	- "aabb" -> "a"
	- "aabb" -> "aa"
	- "aabb" -> "aab"
	- "aabb" -> "aabb"
	- "aabb" -> "a"
	- "aabb" -> "abb"
	- "aabb" -> "b"
	- "aabb" -> "bb"
	- "aabb" -> "b"


Example 3:

	Input: word = "he"
	Output: 2
	Explanation: The two wonderful substrings are underlined below:
	- "he" -> "h"
	- "he" -> "e"

	

Note:

	1 <= word.length <= 10^5
	word consists of lowercase English letters from 'a' to 'j'.


### 解析

根据题意，美妙的字符串是指最多一个字母出现奇数次的字符串。例如，“ccjjc” 和 “abab”是美妙的，但 “ab” 不是。给定一个由前十个小写英文字母（ 'a'  到  'j' ）组成的字符串 word ，返回 word 中美妙的非空子串的数量。 如果相同的子字符串在 word 中出现多次，则分别计算每次出现的次数。子字符串是字符串中连续的字符序列。

最暴力的解法肯定是找出所有的子字符串，然后判断每个子字符串是否满足题意，这种肯定是超时的，因为已经到了 O(n^3) 了。

### 解答
				
	
	class Solution(object):
	    def wonderfulSubstrings(self, word):
	        """
	        :type word: str
	        :rtype: int
	        """
	        result = 0
	        n = len(word)
	        for i in range(n):
	            for j in range(i+1,n+1):
	                c = collections.Counter(word[i:j])
	                count = 0
	                for v in c.values():
	                    if v%2==1:
	                        count += 1
	                        if count > 1:
	                            break
	                if count <= 1:
	                    result += 1
	        return result
            	      
			
### 运行结果

	Time Limit Exceeded


### 解析

因为美妙的字符串最多有一个字符是奇数个，所以有两种情况：

* 第一种情况是出现的所有字符都是偶数
* 第二种情况是出现的字符中只有一个字符是奇数个，其余都是偶数个

因为题中给出了字符串最多有 10 种字符，所以使用最低位表示 ‘a’ 最高位表示 ‘j’ 的 10 位 bit 位变量 mask 来表示遍历到某个字符时候，出现的所有字符的奇偶状态， 1 表示奇数，0 表示偶数，如 0000000001 表示遍历到某个位置 'a' 出现的个数为奇数个。

* words[i:j] 如果满足第一种情况，我们只要知道 mask<sub>i</sub> 和 mask<sub>j</sub> （i < j）相等即可，说明 word[:i] 和 word[:j] 中的每个字母的个数是同奇同偶的，这就说明 word[i:j] 中的包含的字母个数都是偶数，我们只要知道这样的 mask 出现过多少次即可。
* words[i:j] 如果满足第二种情况，那么我们只需要让某个字符的出现个数是奇数即可 ，满足 mask<sub>j</sub>=mask<sub>i</sub> ^(1\<\<k) 即可，即 word[:i] 和 word[:j] 中只有第 k 个字符是不同奇同偶的，其他的字符都是同奇同偶的，这种情况只有一个字符是奇数次数，我们知道这样的 mask 出现过多少次即可。


### 解答

	class Solution(object):
	    def wonderfulSubstrings(self, word):
	        """
	        :type word: str
	        :rtype: int
	        """
	        mask = 0
	        prefix = defaultdict(int)
	        prefix[0] += 1
	        ans = 0
	        for w in word:
	            mask ^= 1 << (ord(w) - ord('a'))
	            ans += prefix[mask]
	            for i in range(10):
	                tmp = mask ^ (1 << i)
	                ans += prefix[tmp]
	            prefix[mask] += 1
	        return ans

### 运行结果

	Runtime: 2444 ms, faster than 30.44% of Python online submissions for Number of Wonderful Substrings.
	Memory Usage: 15 MB, less than 52.17% of Python online submissions for Number of Wonderful Substrings.

原题链接：https://leetcode.com/problems/number-of-wonderful-substrings/



您的支持是我最大的动力
