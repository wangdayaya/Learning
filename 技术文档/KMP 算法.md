## 介绍
KMP 算法是从字符串快速找出某个子字符串的匹配算法，由 D.E.Knuth ，J.H.Morris 和 V.R.Pratt 提出的，所以简称为 KMP 算法。

## 暴力的字符串匹配算法
假如有一个主串 abcd 和目标串 cd ，我们可以暴力进行匹配。

第一次按两个字符串对应位置比较：发现 a==a ，但是 b!=c ，所以不匹配。

	a b c d 
	c d
将 ac 向右移动以为，第二次按两个字符串对应位置比较，发现一开始 b!=a ，所以不匹配：
	
	a b c d
	  c d
再将 ac 向右移动一位，第三次按两个字符串对应位置比较，发现每一位都一样 ，所以完成匹配：
	
	a b c d
	    c d
    

这种暴力的匹配方法时间复杂度比较高，有 O(m*n) ，m 和 n 分别为两个字符串的长度。

## 改良的匹配算法的思想
其实对于上面的匹配方法，主要的耗时操作不在于对应字符的比较，而在于当前字符不匹配的情况下，得向右移动一位并重复比较过程，而改良的匹配算法 KMP 通过找出目标串的最长的公共前后缀，然后可以根据这个最长的公共前后缀将目标串直接向右移动若干位。这样就能将时间复杂度压缩为 O(m+n) 。所以 KMP 算法的核心就是在找当字符比较失败之后，目标串最大的右移位数。这里用 ABCAC 和 ABCAB 两个字符串直观展示找出最大的右移位数的优势。

开始按对应位置比较字符，发现 C!=B 

	A B C A C
	A B C A B
	
假如我们按照暴力解决，接下来肯定会像上面一样将 ABCAB 右移一位，进行比较，发现不匹配，再将 ABCAB 右移一位进行下去。

现在我们可以直接找出 ABCAC 和 ABCAB 相同的前缀为 ABCA ，而 ABCA 的最长公共前后缀为 A ，那么我们当比较到 C!=B 的时候，可以直接按照下面方式右移，此时我们至少知道两个字符串已经有第一个字符 A 是相等的，直接从主串的 C 和目标串的 B 开始比较即可：

	A B C A C
	      A B C A B

用暴力算法我们也可以将主串和目标串移动到这种样子，但是却需要很多步，而用 KMP 此时我们发现可以省略掉暴力算法的很多步骤。

## 详细过程

假如我们有主串  ABCACAAABA 和目标串 ABCABF ，首先我们找出 ABCABF 不同索引下的最长公共前后缀长度，如下：

|  模式  | 前缀  | 后缀 | 最长公共前后缀长度 | 
|  ----  | ----  |----  | ----  |
| A | -|-  | 0 |
| AB | A |B | 0 |
| ABC  | A,AB |BC,C  | 0 |
| ABCA  | A,AB,ABC |BCA,CA,A  |1(A) |
| ABCAB | A,AB,ABC,ABCA |BCAB,CAB,AB,B  | 2(AB) |
| ABCABF | A,AB,ABC,ABCA,ABCAB |BCABF,CABF,ABF,BF,F |0 |


一开始将两个字符串进行比较，发现 C!=B ，要将：

	ABCACAAABA
	ABCABF	

已经匹配到的是 ABCA ，根据我们刚才计算出来的表，发现 ABCA 的最长公共前后缀为 A ，因为 ABCA 长度为 4，A 长度为 1，所以将 ABCABF 右移 3 位，因为 A 已经是相同的，所以接下来直接比较主串的 C 和目标串的 B：

	ABCACAAABA
	   ABCABF	
	  
比较发现 C!=B ，因为目前主串和目标串的共有部分都是 A ，根据上表可以知道，最长公共前后缀为 0 ，所以此时只能右移一位：

	ABCACAAABA
	    ABCABF	

比较发现 C!=A ，因为此时主串和目标串还没有共有部分，所以还是只能右移一位：

	ABCACAAABA
	     ABCABF	
	     
按照上面的过程重新比较即可。

## python 实现

上面的表的发现过程就是 next 函数，这个函数是关键。next[i] 中记录的就是当索引为 i 的时候的子字符串的最长公共前后缀的长度。如字符串 ababc 所构造出来的 next  为 [0, 0, 1, 2, 0] 。next[2] = 2，也就是子字符串 abab 的最长公共前后缀为 2 。

	def KMP_algorithm(s, target):
	    next = getNext(target)
	    n = len(s)
	    m = len(target)
	    i, j = 0, 0
	    while (i < n) and (j < m):
	        if (s[i] == target[j]):
	            i += 1
	            j += 1
	        elif (j != 0):
	            j = next[j - 1]
	        else:
	            i += 1
	    if (j == m):
	        return i - j
	    else:
	        return -1
	
	
	def getNext(s):
	    index= 0 # 开始进行比较的前缀的末尾字符的索引
	    i = 1 #  用来遍历字符串 s
	    m = len(s)
	    next = [0] * m
	    while i < m:
	        if (s[i] == s[index]):
	            next[i] = index + 1 # index + 1 表示最长公共前后缀的长度
	            index += 1
	            i += 1
	        elif (index != 0):
	            index = next[index - 1] # 关键一步，当 s[i] != s[index] 且 index 不为 0 的时候，index 更新为前一个 index 位置上的值
	        else:
	            next[i] = 0
	            i += 1
	    return next
	
	print(KMP_algorithm('abcxabcdabcdabcy', 'abababc'))

## 参考

* https://www.bilibili.com/video/BV1eq4y1X73t（墙裂推荐，说重点非常清楚）
* https://blog.csdn.net/weixin_39561100/article/details/80822208
