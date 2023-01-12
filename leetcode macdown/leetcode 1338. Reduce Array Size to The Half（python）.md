leetcode  1338. Reduce Array Size to The Half（python）

### 描述


You are given an integer array arr. You can choose a set of integers and remove all the occurrences of these integers in the array.

Return the minimum size of the set so that at least half of the integers of the array are removed.


Example 1:

	Input: arr = [3,3,3,3,5,5,5,2,2,7]
	Output: 2
	Explanation: Choosing {3,7} will make the new array [5,5,5,2,2] which has size 5 (i.e equal to half of the size of the old array).
	Possible sets of size 2 are {3,5},{3,2},{5,2}.
	Choosing set {2,7} is not possible as it will make the new array [3,3,3,3,5,5,5] which has size greater than half of the size of the old array.

	
Example 2:

	Input: arr = [7,7,7,7,7,7]
	Output: 1
	Explanation: The only possible set you can choose is {7}. This will make the new array empty.


Example 3:

	Input: arr = [1,9]
	Output: 1

	
Example 4:

	Input: arr = [1000,1000,3,7]
	Output: 1

	
Example 5:

	Input: arr = [1,2,3,4,5,6,7,8,9,10]
	Output: 5


Note:

	1 <= arr.length <= 10^5
	arr.length is even.
	1 <= arr[i] <= 10^5


### 解析


根据题意，给定一个整数数组 arr。 可以从中选择一组整数并删除数组中出现过的所有这些整数。题目要求我们返回最小集合的大小，以便删除数组中至少一半的整数。看题目中的条件限制有点懵，已经明确告诉 arr.length 是偶数，但是又说 arr.length 大于等于 1 小于等于 10^5 ，are you kiding ？是我数学没学好吗 1 怎么能是偶数呢？

思路比较简单：

* 初始化结果 result 为 0 ，L 为 arr 长度的一半，使用内置函数 Counter 对 arr 进行计数
* 对计数结果按照 value 降序排列，遍历所有的键值对，只要 L 还大于 0 就 result 加一，同时 L 减 value ，循环这个过程直到 break ，这样能移除最少的数字实现删除一半以上的整数
* 最后返回 result

### 解答
				

	class Solution(object):
	    def minSetSize(self, arr):
	        """
	        :type arr: List[int]
	        :rtype: int
	        """
	        c = collections.Counter(arr)
	        L = len(arr)//2
	        result = 0
	        for k,v in sorted(c.items(), key=lambda x:-x[1]):
	            if L>0:
	                result += 1
	                L -= v
	            else:
	                break
	        return result
	                
            	      
			
### 运行结果

	Runtime: 640 ms, faster than 52.86% of Python online submissions for Reduce Array Size to The Half.
	Memory Usage: 37.3 MB, less than 35.71% of Python online submissions for Reduce Array Size to The Half.



原题链接：https://leetcode.com/problems/reduce-array-size-to-the-half/



您的支持是我最大的动力
