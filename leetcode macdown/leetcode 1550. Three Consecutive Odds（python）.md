leetcode  1550. Three Consecutive Odds（python）

### 描述


Given an integer array arr, return true if there are three consecutive odd numbers in the array. Otherwise, return false.
 


Example 1:

	
	Input: arr = [2,6,4,1]
	Output: false
	Explanation: There are no three consecutive odds.
	
Example 2:

	Input: arr = [1,2,34,3,4,5,7,23,12]
	Output: true
	Explanation: [5,7,23] are three consecutive odds.




Note:


	1 <= arr.length <= 1000
	1 <= arr[i] <= 1000

### 解析


根据题意，就是判断是否 arr 中存在三个相连的奇数。直接遍历 arr ，用 count 记录出现的奇数个数，如果是奇数则加一，如果不是奇数则重新置为 0 ，如果 count 大于等于 3 的时候直接返回 True ，否则最后遍历结束直接返回 False 。

### 解答
				
	class Solution(object):
	    def threeConsecutiveOdds(self, arr):
	        """
	        :type arr: List[int]
	        :rtype: bool
	        """
	        count = 0
	        for i,v in enumerate(arr):
	            if v%2:
	                count+=1
	                if count==3:
	                    return True
	            else:
	                count = 0
	        return False

            	      
			
### 运行结果


	Runtime: 32 ms, faster than 67.58% of Python online submissions for Three Consecutive Odds.
	Memory Usage: 13.4 MB, less than 91.78% of Python online submissions for Three Consecutive Odds.


### 解析

上面这种思路比较中规中矩，一般人看完题目之后都能想到，下面这种虽然原理是一样的，但是解法比较新奇，因为奇数的特性就是对 2 取模得 1 ，而 3 个相连的奇数对 2 取模就要出现 3 个 1 ，将 arr 中的每个数都取模得到的结果变成字符串连接成字符串，然后只需要判断字符串 111 是否在其中即可。

### 解答

	class Solution(object):
	    def threeConsecutiveOdds(self, arr):
	        """
	        :type arr: List[int]
	        :rtype: bool
	        """
	        return "111" in "".join([str(i%2) for i in arr])

### 运行结果

	Runtime: 36 ms, faster than 36.54% of Python online submissions for Three Consecutive Odds.
	Memory Usage: 13.4 MB, less than 67.31% of Python online submissions for Three Consecutive Odds.


### 解析
这种算法是用到了比较少见的按位与运算 & ，因为某数字与 1 进行按位与运算的时候，当该数字是奇数的时候计算结果为 1 ，当该数字是偶数的时候计算结果为 0 ，所以可以沿用上面的思路，先对 arr 中的每个元素都对 1 做按位与运算，然后将得到的结果拼接为字符串，只需要判断字符串 111 是否在其中出现即可。

### 解答

	class Solution(object):
	    def threeConsecutiveOdds(self, arr):
	        """
	        :type arr: List[int]
	        :rtype: bool
	        """
	        return '111' in ''.join([str(i&1) for i in arr])
	        
### 运行结果

	Runtime: 36 ms, faster than 36.54% of Python online submissions for Three Consecutive Odds.
	Memory Usage: 13.5 MB, less than 67.31% of Python online submissions for Three Consecutive Odds.

原题链接：https://leetcode.com/problems/three-consecutive-odds/



您的支持是我最大的动力
