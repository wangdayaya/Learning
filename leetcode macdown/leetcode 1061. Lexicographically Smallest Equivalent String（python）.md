leetcode  1061. Lexicographically Smallest Equivalent String（python）




### 描述


You are given two strings of the same length s1 and s2 and a string baseStr. We say s1[i] and s2[i] are equivalent characters.

* For example, if s1 = "abc" and s2 = "cde", then we have 'a' == 'c', 'b' == 'd', and 'c' == 'e'.

Equivalent characters follow the usual rules of any equivalence relation:

* Reflexivity: 'a' == 'a'.
* Symmetry: 'a' == 'b' implies 'b' == 'a'.
* Transitivity: 'a' == 'b' and 'b' == 'c' implies 'a' == 'c'.

For example, given the equivalency information from s1 = "abc" and s2 = "cde", "acd" and "aab" are equivalent strings of baseStr = "eed", and "aab" is the lexicographically smallest equivalent string of baseStr. Return the lexicographically smallest equivalent string of baseStr by using the equivalency information from s1 and s2.


Example 1:

	Input: s1 = "parker", s2 = "morris", baseStr = "parser"
	Output: "makkek"
	Explanation: Based on the equivalency information in s1 and s2, we can group their characters as [m,p], [a,o], [k,r,s], [e,i].
	The characters in each group are equivalent and sorted in lexicographical order.
	So the answer is "makkek".

	
Example 2:

	Input: s1 = "hello", s2 = "world", baseStr = "hold"
	Output: "hdld"
	Explanation: Based on the equivalency information in s1 and s2, we can group their characters as [h,w], [d,e,o], [l,r].
	So only the second letter 'o' in baseStr is changed to 'd', the answer is "hdld".


Example 3:

	Input: s1 = "leetcode", s2 = "programs", baseStr = "sourcecode"
	Output: "aauaaaaada"
	Explanation: We group the equivalent characters in s1 and s2 as [a,o,e,r,s,c], [l,p], [g,t] and [d,m], thus all letters in baseStr except 'u' and 'd' are transformed to 'a', the answer is "aauaaaaada".




Note:

	1 <= s1.length, s2.length, baseStr <= 1000
	s1.length == s2.length
	s1, s2, and baseStr consist of lowercase English letters.


### 解析


根据题意，给定两个长度相同的字符串 s1 和 s2 以及一个字符串 baseStr。我们认为 s1[i] 和 s2[i] 是等效的字符。

* 例如，如果 s1 = “abc” 且 s2 = “cde”，那么我们有 'a' == 'c'， 'b' == 'd' 和 'c' == 'e'。

等价字符遵循任何等价关系的常用规则：

* 自反性：“a” == “a”。
* 对称性：“a” == “b” 表示 “b” == “a”。
* 传递性：“a” == “b” 和 “b” == “c” 表示 “a” == “c”。

例如，给定 s1 = “abc” 和 s2 = “cde” 的等价信息，“acd” 和 “aab” 是 baseStr = “eed” 的等价字符串，而 “aab” 是 baseStr 的字典顺序最小的等价字符串。通过使用 s1 和 s2 中的等价信息返回 baseStr 在字典顺序的最小等效字符串。

这道题的关键在于对应位置的字符要满足传递性和对称性，因为自反性本身就可以通过对字符串的判断检查出来。题目要满足字典顺序最小的答案，所以每个字母肯定能找到一个与其对应的字典序最小的字母，这时候我们只需要针对传递性和对称性，结合并查集算法来找字典序最小的“祖先字母”即可。

* 我们首先定义 0-25 表示 a-z ，初始化一个长度为 26 的数组 roots ，每个位置表示从 a-z 每个字母的“祖先字母”。
* 同时遍历 s1 和 s2 相同位置上的字母，找到各自的“祖先字母”，如果有较小的“祖先字母”出现，就将他们两个的“祖先字母”进行更新。
* 遍历结束，roots 每个位置上存放的就是 a-z 每个字母对应的祖先字母与 'a' 的 ascii 码距离，然后遍历 baseStr ，将每个字符进行对应转化即可得到结果。

时间复杂度为 O(N \* 26 + M \* 26) ，N 为 s1 或 s2 的长度，M 是 baseStr 的长度 ，在进行并查集搜索的时候可能最多需要 26 次，空间复杂度为 O(26) 。

### 解答
				

	class Solution:
	    def smallestEquivalentString(self, s1: str, s2: str, baseStr: str) -> str:
	        def find(c):
	            while roots[c] != c:
	                c = roots[c]
	            return c
	        roots = list(range(26))
	        for c1, c2 in zip(s1, s2):
	            r1 = find(ord(c1) - ord('a'))
	            r2 = find(ord(c2) - ord('a'))
	            if r1 > r2:
	                r1, r2 = r2, r1
	            roots[r2] = r1
	        return "".join(chr(ord('a') + find(ord(c) - ord('a'))) for c in baseStr)
            	      
			
### 运行结果

	Runtime 39 ms ，Beats 93.68%
	Memory 13.8 MB ，Beats 100%


原题链接：https://leetcode.com/problems/lexicographically-smallest-equivalent-string/



您的支持是我最大的动力
