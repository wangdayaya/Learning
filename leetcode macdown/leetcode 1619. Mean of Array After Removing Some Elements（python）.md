leetcode  1619. Mean of Array After Removing Some Elements（python）

### 描述



	Given an integer array arr, return the mean of the remaining integers after removing the smallest 5% and the largest 5% of the elements.
	
	Answers within 10-5 of the actual answer will be considered accepted.
	
	 

Example 1:

	Input: arr = [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3]
	Output: 2.00000
	Explanation: After erasing the minimum and the maximum values of this array, all elements are equal to 2, so the mean is 2.

	
Example 2:

	Input: arr = [6,2,7,5,1,2,0,3,10,2,5,0,5,5,0,8,7,6,8,0]
	Output: 4.00000


Example 3:


	Input: arr = [6,0,7,0,7,5,7,8,3,4,0,7,8,1,6,8,1,1,2,4,8,1,9,5,4,3,8,5,10,8,6,6,1,0,6,10,8,2,3,4]
	Output: 4.77778
	
Example 4:

	Input: arr = [9,7,8,7,7,8,4,4,6,8,8,7,6,8,8,9,2,6,0,0,1,10,8,6,3,3,5,1,10,9,0,7,10,0,10,4,1,10,6,9,3,6,0,0,2,7,0,6,7,2,9,7,7,3,0,1,6,1,10,3]
	Output: 5.27778

	
Example 5:

	Input: arr = [4,8,4,10,0,7,1,3,7,8,8,3,4,1,6,2,1,1,8,0,9,8,0,3,9,10,3,10,1,10,7,3,2,1,4,9,10,7,6,4,0,8,5,1,2,1,6,2,5,0,7,10,9,10,3,7,10,5,8,5,7,6,7,6,10,9,5,10,5,5,7,2,10,7,7,8,2,0,1,1]
	Output: 5.29167


Note:


20 <= arr.length <= 1000
arr.length is a multiple of 20.
0 <= arr[i] <= 10^5

### 解析

根据题意，计算 arr 中去掉最小的和最大的 5% 的元素的平均数，先进行排序，然后计算出长度的 5% 的大小，然后直接在 arr 列表上进行截取，然后计算平均值即可。


### 解答
				
	class Solution(object):
	    def trimMean(self, arr):
	        """
	        :type arr: List[int]
	        :rtype: float
	        """
	        arr.sort()
	        N = int(len(arr) * 0.05)
	        result = arr[N:-N]
	        return float(sum(result))/len(result)

            	      
			
### 运行结果

	Runtime: 40 ms, faster than 77.37% of Python online submissions for Mean of Array After Removing Some Elements.
	Memory Usage: 13.5 MB, less than 71.53% of Python online submissions for Mean of Array After Removing Some Elements.


### 解析

上面的代码是直接截取了列表的部分进行计算，这里我们也可以将 arr 进行排序，然后遍历 %5 长度的次数，然后每次都去掉最前面和最后面的元素，再将剩下的元素求平均即可得到答案。

### 解答
					
	class Solution(object):
	    def trimMean(self, arr):
	        """
	        :type arr: List[int]
	        :rtype: float
	        """
	        arr.sort()
	        for _ in range(int(len(arr) * .05)):
	            arr.pop(0)
	            arr.pop()
	        return float(sum(arr)) / len(arr)
            	      
			
### 运行结果

	Runtime: 32 ms, faster than 99.15% of Python online submissions for Mean of Array After Removing Some Elements.
	Memory Usage: 13.8 MB, less than 19.66% of Python online submissions for Mean of Array After Removing Some Elements.

原题链接：https://leetcode.com/problems/mean-of-array-after-removing-some-elements/



您的支持是我最大的动力
