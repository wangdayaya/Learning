leetcode  2300. Successful Pairs of Spells and Potions（python）


本题是 Biweekly Contest 80  的第二道题，难度为 Medium ，题目主要考察的内容是排序和二分法的常规操作。

### 描述

You are given two positive integer arrays spells and potions, of length n and m respectively, where spells[i] represents the strength of the ith spell and potions[j] represents the strength of the jth potion.

You are also given an integer success. A spell and potion pair is considered successful if the product of their strengths is at least success.

Return an integer array pairs of length n where pairs[i] is the number of potions that will form a successful pair with the ith spell.



Example 1:


	Input: spells = [5,1,3], potions = [1,2,3,4,5], success = 7
	Output: [4,0,3]
	Explanation:
	- 0th spell: 5 * [1,2,3,4,5] = [5,10,15,20,25]. 4 pairs are successful.
	- 1st spell: 1 * [1,2,3,4,5] = [1,2,3,4,5]. 0 pairs are successful.
	- 2nd spell: 3 * [1,2,3,4,5] = [3,6,9,12,15]. 3 pairs are successful.
	Thus, [4,0,3] is returned.
	
Example 2:


	Input: spells = [3,1,2], potions = [8,5,8], success = 16
	Output: [2,0,2]
	Explanation:
	- 0th spell: 3 * [8,5,8] = [24,15,24]. 2 pairs are successful.
	- 1st spell: 1 * [8,5,8] = [8,5,8]. 0 pairs are successful. 
	- 2nd spell: 2 * [8,5,8] = [16,10,16]. 2 pairs are successful. 
	Thus, [2,0,2] is returned.






Note:

	n == spells.length
	m == potions.length
	1 <= n, m <= 10^5
	1 <= spells[i], potions[i] <= 10^5
	1 <= success <= 10^10


### 解析

根据题意，给定两个正整数数组 spell 和 potion ，长度分别为 n 和 m ，其中 spells[i] 代表第 i 个法术的强度 ，potions[j] 代表第 j 个药水的强度 。题目还给定一个整数 success 。 如果 spells[i] 与 potions[j] 的乘大于等于 success ，那么咒语和药水配对被认为是成功的。 返回一个长度为 n 的整数数组 pairs ，其中 pairs[i] 是与第 i 个法术成功配对的药水数量。

这道题其实就是考查二分法的常规用法，我们已知了 spells ，要求能与  spells[i]  成功配对的药水数量，所以我们将 potions 先进行升序的排序，然后我们遍历 spells[i]  ，求出符合要求的药水强度的最小值为 math.ceil(success/spells[i]) ，所以我们只需要在  potions 中通过二分法查找大于等于该药水强度的数量，并将其加入到 result 即可，不断重复上面的操作，最后得到的 result 即为结果。

时间复杂度为 O(NlogN) ，空间复杂度为 O(N) 。

### 解答
				

	class Solution:
	    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
	        N = len(potions)
	        potions.sort()
	        result = []
	        for s in spells:
	            target = math.ceil(success/s)
	            idx = bisect.bisect_left(potions, target)
	            result.append(N-idx)
	        return result
            	      
			
### 运行结果

	

	56 / 56 test cases passed.
	Status: Accepted
	Runtime: 1477 ms
	Memory Usage: 37.1 MB

### 原题链接

https://leetcode.com/contest/biweekly-contest-80/problems/successful-pairs-of-spells-and-potions/

您的支持是我最大的动力
