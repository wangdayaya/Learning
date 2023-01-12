leetcode  2060. Check if an Original String Exists Given Two Encoded Strings（python）

### 描述

An original string, consisting of lowercase English letters, can be encoded by the following steps:

* Arbitrarily split it into a sequence of some number of non-empty substrings.
* Arbitrarily choose some elements (possibly none) of the sequence, and replace each with its length (as a numeric string).
* Concatenate the sequence as the encoded string.

For example, one way to encode an original string "abcdefghijklmnop" might be:

* Split it as a sequence: ["ab", "cdefghijklmn", "o", "p"].
* Choose the second and third elements to be replaced by their lengths, respectively. The sequence becomes ["ab", "12", "1", "p"].
* Concatenate the elements of the sequence to get the encoded string: "ab121p".

Given two encoded strings s1 and s2, consisting of lowercase English letters and digits 1-9 (inclusive), return true if there exists an original string that could be encoded as both s1 and s2. Otherwise, return false.

Note: The test cases are generated such that the number of consecutive digits in s1 and s2 does not exceed 3.



Example 1:

	Input: s1 = "internationalization", s2 = "i18n"
	Output: true
	Explanation: It is possible that "internationalization" was the original string.
	- "internationalization" 
	  -> Split:       ["internationalization"]
	  -> Do not replace any element
	  -> Concatenate:  "internationalization", which is s1.
	- "internationalization"
	  -> Split:       ["i", "nternationalizatio", "n"]
	  -> Replace:     ["i", "18",                 "n"]
	  -> Concatenate:  "i18n", which is s2

	
Example 2:

	Input: s1 = "l123e", s2 = "44"
	Output: true
	Explanation: It is possible that "leetcode" was the original string.
	- "leetcode" 
	  -> Split:      ["l", "e", "et", "cod", "e"]
	  -> Replace:    ["l", "1", "2",  "3",   "e"]
	  -> Concatenate: "l123e", which is s1.
	- "leetcode" 
	  -> Split:      ["leet", "code"]
	  -> Replace:    ["4",    "4"]
	  -> Concatenate: "44", which is s2.


Example 3:

	Input: s1 = "a5b", s2 = "c5b"
	Output: false
	Explanation: It is impossible.
	- The original string encoded as s1 must start with the letter 'a'.
	- The original string encoded as s2 must start with the letter 'c'.

	
Example 4:
	
	Input: s1 = "112s", s2 = "g841"
	Output: true
	Explanation: It is possible that "gaaaaaaaaaaaas" was the original string
	- "gaaaaaaaaaaaas"
	  -> Split:      ["g", "aaaaaaaaaaaa", "s"]
	  -> Replace:    ["1", "12",           "s"]
	  -> Concatenate: "112s", which is s1.
	- "gaaaaaaaaaaaas"
	  -> Split:      ["g", "aaaaaaaa", "aaaa", "s"]
	  -> Replace:    ["g", "8",        "4",    "1"]
	  -> Concatenate: "g841", which is s2.


	
Example 5:

	Input: s1 = "ab", s2 = "a2"
	Output: false
	Explanation: It is impossible.
	- The original string encoded as s1 has two letters.
	- The original string encoded as s2 has three letters.



Note:


	1 <= s1.length, s2.length <= 40
	s1 and s2 consist of digits 1-9 (inclusive), and lowercase English letters only.
	The number of consecutive digits in s1 and s2 does not exceed 3.

### 解析

根据题意，一个由小写英文字母组成的原始字符串可以通过以下步骤进行编码：

* 将其任意拆分为若干个非空子串
* 任意选择非空子串（可能不选），并将每个其替换为其长度，作为数字字符串
* 将变化后的所有子字符串连接为编码字符串

题目比较晦涩，但是可以结合例子就可以明白了。例如，编码原始字符串 “abcdefghijklmnop” 的一种方法可能是：

* 将其拆分为一个序列：["ab", "cdefghijklmn", "o", "p"]。
* 随意选择要由其长度替换的第二个和第三个元素。序列变为 ["ab", "12", "1", "p"]。
* 连接结果连接以获得编码字符串：“ab121p”。

给定两个编码字符串 s1 和 s2，由小写英文字母和数字 1-9（含）组成，如果存在可以同时编码为 s1 和 s2 的原始字符串，则返回 True 。否则返回 False 。

这道题的关键在于中间的数字字符串怎么拆解，如 “ab121a” ，会有四种分解情况：

* 可以表示有长度为 1 、2 、1 的三个子字符串
* 可以表示有长度为 12 、 1 的两个子字符串
* 可能是 1 、21 的两个子字符串
* 也可能是一个长度为 121 的子字符串

通过看限制条件，我们发现给出的字符串 s1 和 s2 长度不超过 40 ，而且 s1 和 s2 中连续数字的数量不超过 3 ，那么按照最坏的情况，字符串中每 3 个数字都会被 1 个字母隔开，那就是一共有 40//4 个区间，而每个区间的 3 个数字和 1 个字母会有像上面提到的四种情况，那一共会有 4^10 ，说明该题是可以暴力求解的。使用递归来分解数字字符串，同时用记忆集合保存已经经过的递归，这样可以提速。


### 解答
				
	class Solution(object):
	    def possiblyEquals(self, s1, s2):
	        """
	        :type s1: str
	        :type s2: str
	        :rtype: bool
	        """
	        s1 = self.parse(s1)
	        s2 = self.parse(s2)
	        visited = set()
	        def dfs(s1, i, n1, s2, j, n2):
	            if i==len(s1) and j==len(s2): return n1==n2
	            if i==len(s1) and n1==0 : return False
	            if j==len(s2) and n2==0 : return False
	            h = i*1e9 + n1*1e6+ j*1e3+ n2;
	            if (h in visited): return False;
	            if i<len(s1) and s1[i][0].isdigit():
	                nums = self.getNum(s1[i]) 
	                for x in nums:
	                    if dfs(s1, i+1, n1+x, s2 , j , n2):
	                        return True
	                visited.add(h)
	                return False
	            elif j<len(s2) and s2[j][0].isdigit():
	                nums = self.getNum(s2[j]) 
	                for x in nums:
	                    if dfs(s1, i, n1, s2 , j+1 , n2+x):
	                        return True
	                visited.add(h)
	                return False
	            if n1!=0 and n2!=0:
	                common = min(n1, n2)
	                return dfs(s1, i, n1-common, s2, j, n2-common)
	            elif n1!=0 and n2==0:
	                return dfs(s1, i, n1-1, s2, j+1, 0)
	            elif n1==0 and n2!=0:
	                return dfs(s1, i+1, 0, s2, j, n2-1)
	            else:
	                visited.add(h)
	                if s1[i]!=s2[j]: return False
	                return dfs(s1, i+1, 0, s2, j+1, 0)
	            
	        return dfs(s1, 0, 0, s2, 0, 0)
	    
	    def getNum(self, s):
	        n = int(s)
	        if len(s) == 1: return {n}
	        elif len(s) == 2:
	            a = n/10
	            b = n%10
	            return {a+b, n}
	        elif len(s) == 3:
	            a = n/100
	            b = (n/10)%10
	            c = n%10
	            return {a+b+c, a+b*10+c, a*10+b+c , n}
	            
	            
	    def parse(self, s):
	        t = []
	        i = 0
	        while i<len(s):
	            if s[i].isalpha():
	                t.append(s[i:i+1])
	            else:
	                j = i
	                while j<len(s) and s[j].isdigit():
	                    j += 1
	                t.append(s[i:j])
	                i = j-1
	            i += 1
	        return t
	                
	        
	                
	        

            	      
			
### 运行结果

	Runtime: 2131 ms, faster than 35.37% of Python online submissions for Check if an Original String Exists Given Two Encoded Strings.
	Memory Usage: 30.5 MB, less than 7.32% of Python online submissions for Check if an Original String Exists Given Two Encoded Strings.

原题链接：https://leetcode.com/problems/check-if-an-original-string-exists-given-two-encoded-strings/



您的支持是我最大的动力
