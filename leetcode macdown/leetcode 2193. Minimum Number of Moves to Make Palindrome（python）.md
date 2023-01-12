leetcode 2193. Minimum Number of Moves to Make Palindrome （python）


### 前言

这是 Biweekly Contest 73 比赛的第四题，主要考察的就是回文概念和贪心算法，难度为 Hard 。


### 描述


You are given a string s consisting only of lowercase English letters.

In one move, you can select any two adjacent characters of s and swap them.

Return the minimum number of moves needed to make s a palindrome.

Note that the input will be generated such that s can always be converted to a palindrome.

 


Example 1:


	Input: s = "aabb"
	Output: 2
	Explanation:
	We can obtain two palindromes from s, "abba" and "baab". 
	- We can obtain "abba" from s in 2 moves: "aabb" -> "abab" -> "abba".
	- We can obtain "baab" from s in 2 moves: "aabb" -> "abab" -> "baab".
	Thus, the minimum number of moves needed to make s a palindrome is 2.
	


Note:


	1 <= s.length <= 2000
	s consists only of lowercase English letters.
	s can be converted to a palindrome using a finite number of moves.

### 解析

根据题意，给定一个仅由小写英文字母组成的字符串 s。使用一次操作 move 可以选择 s 的任意两个相邻的字符并交换它们。返回使 s 成为回文所需的最小移动次数。题目中已经保证了输入的 s 肯定能形成回文。

这道题我比赛的时候没做出来，之后看了些大佬的解答，有两种解法比较好理解，先介绍第一种，假如我们现在有一个 s 为 lceetelet ：

* 我们以第一个字母 l 为基准，从最后一个字符往前找 l ，找到之后使用了两次操作变换，经过交换字符得到 lceeteetl 。
* 然后我们以第二个字母 c 为基准，从倒数第二个字符开始向前找（因为倒数第一个已经在刚才固定了是 l ），因为 c 只有一个，所以从后往前找只能找到第二个字符 c ，说明这个字符是要把它放到正中间的，操作的次数就是 len(s)//2-1，此时暂且不去管它
* 我们以第三个字母 e 为基准，从倒数第二个字符开始向前找，找到之后只使用了一次操作变化，交换之后得到是 lceetetel 
* 我们以第四个字母 e 为基准，从倒数第三个字符开始向前找，找到之后只使用了一次操作变化，交换之后得到 lceetteel 
* 此时都已经找完了，我们只需要把单独的 c 放到中间即可，这样就能得到 leetcteel 。总共消耗了 7 次操作。

时间复杂度为 O(N^2 * CONCAT) ，CONCAT 主要是指代码中有一个步骤是对字符串进行拼接，比较耗时，空间复杂度为O(N)。


### 解答
				

	class Solution(object):
	    def minMovesToMakePalindrome(self, s):
	        result = 0
	        N = len(s)
	        target = N - 1 
	        for i in range(N//2):
	            for j in range(target, i-1, -1):
	                if j == i:
	                    result += N // 2 - i
	                elif s[i] == s[j]:
	                    s = s[:j] + s[j + 1 : target + 1] + s[j] + s[target + 1:]
	                    result += target - j
	                    target -= 1
	                    break
	        return result
	        
            	      
			
### 运行结果


	
	129 / 129 test cases passed.
	Status: Accepted
	Runtime: 1027 ms
	Memory Usage: 13.6 MB

### 解析

另外一个大佬的思路更加简单，就是使用贪心的算法，本质上和上面的思路是一样的，就是不断确定字符串第一个和最后一个的字符，然后往中间去确定第二个和倒数第二个字符，以此类推。

而这个大佬的解法比上面的简化了不少。尽管时间复杂度和空间复杂度同上，但是细微之处还是有区别。因为上面有一个步骤是对字符串进行拼接，所以是比较耗时很久，运行了 1027 ms ，而这个解法是纯数字运算，所以耗时很短只有 83 ms 。而且在 while 循环过程中由于两头的元素不断弹出， s 的长度也随之越来越短，在执行 index 的时候时间越来越短，这也是本解法耗时少的一个原因。这位大佬真的是厉害，解法巧妙之极。

### 解答

	class Solution(object):
	    def minMovesToMakePalindrome(self, s):
	        result = 0
	        s = list(s)
	        res = 0
	        while s:
	            i = s.index(s[-1])
	            if i == len(s) - 1:
	                res += i / 2
	            else:
	                res += i
	                s.pop(i)
	            s.pop()
	        return res

### 运行结果

	
	129 / 129 test cases passed.
	Status: Accepted
	Runtime: 83 ms
	Memory Usage: 13.6 MB

### 原题链接



https://leetcode.com/contest/biweekly-contest-73/problems/minimum-number-of-moves-to-make-palindrome/


您的支持是我最大的动力
