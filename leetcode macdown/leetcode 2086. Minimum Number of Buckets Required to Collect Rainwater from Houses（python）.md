leetcode  2086. Minimum Number of Buckets Required to Collect Rainwater from Houses（python）

### 描述

You are given a 0-indexed string street. Each character in street is either 'H' representing a house or '.' representing an empty space.

You can place buckets on the empty spaces to collect rainwater that falls from the adjacent houses. The rainwater from a house at index i is collected if a bucket is placed at index i - 1 and/or index i + 1. A single bucket, if placed adjacent to two houses, can collect the rainwater from both houses.

Return the minimum number of buckets needed so that for every house, there is at least one bucket collecting rainwater from it, or -1 if it is impossible.



Example 1:

	
	Input: street = "H..H"
	Output: 2
	Explanation:
	We can put buckets at index 1 and index 2.
	"H..H" -> "HBBH" ('B' denotes where a bucket is placed).
	The house at index 0 has a bucket to its right, and the house at index 3 has a bucket to its left.
	Thus, for every house, there is at least one bucket collecting rainwater from it.
	
Example 2:

	Input: street = ".H.H."
	Output: 1
	Explanation:
	We can put a bucket at index 2.
	".H.H." -> ".HBH." ('B' denotes where a bucket is placed).
	The house at index 1 has a bucket to its right, and the house at index 3 has a bucket to its left.
	Thus, for every house, there is at least one bucket collecting rainwater from it.


Example 3:

	
	Input: street = ".HHH."
	Output: -1
	Explanation:
	There is no empty space to place a bucket to collect the rainwater from the house at index 2.
	Thus, it is impossible to collect the rainwater from all the houses.
	
Example 4:

	Input: street = "H"
	Output: -1
	Explanation:
	There is no empty space to place a bucket.
	Thus, it is impossible to collect the rainwater from the house.

	
Example 5:

	Input: street = "."
	Output: 0
	Explanation:
	There is no house to collect water from.
	Thus, 0 buckets are needed.



Note:

	1 <= street.length <= 10^5
	street[i] is either'H' or '.'.


### 解析


根据题意，给定一个索引为 0 的字符串 street。 street 中的每个字符要么是代表房子的 'H' ，要么是 '.' 代表一个空的空间。人们可以在空旷的地方放置水桶，以收集从相邻房屋落下的雨水。 如果将水桶放置在索引 i - 1 和/或 索引 i + 1 处，则会收集索引 i 处房屋的雨水。如果将单个水桶放置在两所房屋附近，则可以收集两所房屋的雨水。题目要求返回所需的最小桶数，以便对于每个房屋，至少有一个桶从收集它的雨水，如果不可能，则返回 -1。

读完题目我们可以很直接想到解决这个题要用到贪心策略，我们可以尽量把水桶放到房子的右边的空地上，这样就能尽量少使用桶，思路如下：

* 如果 street 中都是 H 直接返回 -1 ，如果 street 中都是 . 直接返回 0
* 从左到右遍历 street 中的每个字符，如果不是 H 直接跳过进行下一次循环
* 如果是 H 那么如果左边已经有 B ，则直接进行下一次循环；否则如果右边有空地，则将其右边的字符变为 B ；否则如果左边有空地，就将左边的字符变为 B ；否则直接返回 -1 即可
* 最后计算 street 出现了多少个 B 即可

### 解答
				
	class Solution(object):
	    def minimumBuckets(self, street):
	        """
	        :type street: str
	        :rtype: int
	        """
	        if street.count('H') == len(street):return -1
	        if street.count('.') == len(street):return 0
	        street = list(street)
	        for i in range(len(street)):
	            x = street[i]
	            if x == 'H' and i-1>=0 and street[i-1]=='B':
	                continue
	            elif x == 'H' and i+1<len(street) and street[i+1]=='.':
	                street[i+1] = 'B'
	            elif x == 'H' and i-1>=0 and street[i-1]=='.':
	                street[i-1] = 'B'
	            elif x == 'B' or x == '.':
	                continue
	            else:
	                return -1
	        return street.count('B')
	                
	


            	      
			
### 运行结果

	Runtime: 140 ms, faster than 40.22% of Python online submissions for Minimum Number of Buckets Required to Collect Rainwater from Houses.
	Memory Usage: 18.4 MB, less than 30.43% of Python online submissions for Minimum Number of Buckets Required to Collect Rainwater from Houses.

### 解析

当然了我们还可以换一种思路，我们可以通过举例找出下面四种情况出现就一定无法满足题意的情况：

	如果 street 本身就是 H
	如果 street 开头两个字符是 HH
	如果 street 末尾两个字符是 HH
	如果 street 中间的三个字符是 HHH
	
上面四种情况可以直接返回 -1 ，最后因为贪心的最大化的结果就是有多少个 H 就分配几个 B ，然后由于 H.H 这种情况两个房子能共同使用一个水桶，所以使用最少的水桶个数就是 street.count('H')-street.count('H.H') 

### 解答

	class Solution(object):
	    def minimumBuckets(self, street):
	        """
	        :type street: str
	        :rtype: int
	        """
	        return -1 if 'HH'==street[:2] or street[-2:] == 'HH' or 'HHH' in street or street=='H' else street.count('H')-street.count('H.H')
	    

### 运行结果
	
	Runtime: 24 ms, faster than 96.74% of Python online submissions for Minimum Number of Buckets Required to Collect Rainwater from Houses.
	Memory Usage: 14.9 MB, less than 83.70% of Python online submissions for Minimum Number of Buckets Required to Collect Rainwater from Houses.
	
原题链接：https://leetcode.com/problems/minimum-number-of-buckets-required-to-collect-rainwater-from-houses/



您的支持是我最大的动力
