leetcode  2231. Largest Number After Digit Swaps by Parity（python）




### 描述

You are given a positive integer num. You may swap any two digits of num that have the same parity (i.e. both odd digits or both even digits).

Return the largest possible value of num after any number of swaps.



Example 1:

	Input: num = 1234
	Output: 3412
	Explanation: Swap the digit 3 with the digit 1, this results in the number 3214.
	Swap the digit 2 with the digit 4, this results in the number 3412.
	Note that there may be other sequences of swaps but it can be shown that 3412 is the largest possible number.
	Also note that we may not swap the digit 4 with the digit 1 since they are of different parities.


	
Example 2:

	Input: num = 65875
	Output: 87655
	Explanation: Swap the digit 8 with the digit 6, this results in the number 85675.
	Swap the first digit 5 with the digit 7, this results in the number 87655.
	Note that there may be other sequences of swaps but it can be shown that 87655 is the largest possible number.





Note:


1 <= num <= 10^9

### 解析


根据题意，给你一个正整数 num ，题目要求可以交换任何两个奇数或任何两个偶数的任意两位 num 中的数字，可以交换无数次，最后返回变化之后 num 可能的最大值。

比赛的时候看这个题觉得这道题很简单，而自己写的代码有点啰嗦，一直想优化一下代码，但因为事件关系放弃了，比较结束之后看谈论区的各位大佬的代码，也都是又臭又长，才发现他们和我写的是一个水平，瞬间没有了优化代码的欲望。

这道题其实主要考察的就是对数组的基本操作以及简单的排序，思路还是挺简单的：

* 把 num 变为列表 nums ，因为字符串没法进行索引位置上的修改操作
* 把 nums 中的偶数都取出来放入 A 中，然后对 A 进行降序排列
* 将 A 中的元素按照降序顺序，再依次放回 nums 中之前是偶数的位置上
* 把 nums 中的奇数都取出来放入 B 中，然后对 B 进行降序排列
* 将 B 中的元素按照降序顺序，再依次放回 nums 中之前是奇数的位置上
* 将 nums 拼接成字符串返回即可

时间复杂度为 O(N + NlogN) ，空间复杂度为 O(N) 。

### 解答
				
	class Solution(object):
	    def largestInteger(self, num):
	        """
	        :type num: int
	        :rtype: int
	        """
	        nums = list(str(num))
	        N = len(nums)
	        A = [i for i in nums if int(i)%2 == 0]
	        A.sort()
	        A = A[::-1]
	        n = 0
	        for i in range(N):
	            if int(nums[i])%2 == 0:
	                nums[i] = A[n]
	                n += 1
	        B = [i for i in nums if int(i)%2 == 1]
	        B.sort()
	        B = B[::-1]
	        n = 0
	        for i in range(N):
	            if int(nums[i])%2 == 1:
	                nums[i] = B[n]
	                n += 1
	        return int(''.join(nums))

            	      
			
### 运行结果

	
	238 / 238 test cases passed.
	Status: Accepted
	Runtime: 34 ms
	Memory Usage: 13.3 MB


### 原题链接

https://leetcode.com/contest/weekly-contest-288/problems/largest-number-after-digit-swaps-by-parity/

您的支持是我最大的动力
