leetcode  1647. Minimum Deletions to Make Character Frequencies Unique（python）



### 描述


A string s is called good if there are no two different characters in s that have the same frequency.

Given a string s, return the minimum number of characters you need to delete to make s good.

The frequency of a character in a string is the number of times it appears in the string. For example, in the string "aab", the frequency of 'a' is 2, while the frequency of 'b' is 1.




Example 1:

	Input: s = "aab"
	Output: 0
	Explanation: s is already good.

	
Example 2:

	Input: s = "aaabbbcc"
	Output: 2
	Explanation: You can delete two 'b's resulting in the good string "aaabcc".
	Another way it to delete one 'b' and one 'c' resulting in the good string "aaabbc".


Example 3:

	Input: s = "ceabaacb"
	Output: 2
	Explanation: You can delete both 'c's resulting in the good string "eabaab".
	Note that we only care about characters that are still in the string at the end (i.e. frequency of 0 is ignored).

	


Note:

1 <= s.length <= 10^5
s contains only lowercase English letters.


### 解析

根据题意，如果 s 中没有两个具有相同频率的不同字符，则字符串 s 被称为  good string  。给定一个字符串 s，返回您需要删除的最少字符数以使 s 变成 good string 。
一个字符在字符串中出现的频率是它在字符串中出现的次数。 例如，在字符串“aab”中，“a” 的频率为 2，而 “b” 的频率为 1 。

读题之后我们就知道了我们的目标，就是将所有字符出现的次数都变成不一样的最小删除次数，我们在对字符串中的字符进行计数之后，为了要尽可能少的删除次数我们肯定要去操作出现次数少的字符，因为去操作出现次数多的字符虽然结果可能符合 good string 题意，但是肯定不是最少的删除次数。这样我们就要对字符按照出现次数进行升序排序，这样我们按照顺序去处理每个字符，删除其出现的个数直到这个字符的出现数量在所有字符出现数量中没有出现即可。

上面的算法有点贪心思想，其实细想一下不用排序也是一样的，直接对字符出现次数进行删减操作也可以。

时间复杂度为 O(N \* K^2) ，K 为 s 中不同字符的个数，假如最坏情况所有字符的出现次数都相同，那么需要减的次数为 0+1+2+...+K-1 = K\*(K-1) // 2 ，空间复杂度为 O(K) 。

### 解答
				
	class Solution(object):
	    def minDeletions(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        result = 0
	        c = collections.Counter(s)
	        counter = sorted(c.items(), key=lambda x:x[1])
	        for v,n in counter:
	            while c[v] > 0 and list(c.values()).count(c[v]) > 1:
	                c[v] -= 1
	                result += 1
	        return result




            	      
			
### 运行结果

	Runtime: 376 ms, faster than 53.18% of Python online submissions for Minimum Deletions to Make Character Frequencies Unique.
	Memory Usage: 14.4 MB, less than 53.64% of Python online submissions for Minimum Deletions to Make Character Frequencies Unique.


### 原题链接

https://leetcode.com/problems/minimum-deletions-to-make-character-frequencies-unique/

您的支持是我最大的动力
