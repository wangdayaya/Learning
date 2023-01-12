leetcode  496. Next Greater Element I（python）

### 描述


You are given two integer arrays nums1 and nums2 both of unique elements, where nums1 is a subset of nums2.

Find all the next greater numbers for nums1's elements in the corresponding places of nums2.

The Next Greater Number of a number x in nums1 is the first greater number to its right in nums2. If it does not exist, return -1 for this number.

 


Example 1:


	Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
	Output: [-1,3,-1]
	Explanation:
	For number 4 in the first array, you cannot find the next greater number for it in the second array, so output -1.
	For number 1 in the first array, the next greater number for it in the second array is 3.
	For number 2 in the first array, there is no next greater number for it in the second array, so output -1.
	
Example 2:


	Input: nums1 = [2,4], nums2 = [1,2,3,4]
	Output: [3,-1]
	Explanation:
	For number 2 in the first array, the next greater number for it in the second array is 3.
	For number 4 in the first array, there is no next greater number for it in the second array, so output -1.





Note:

	1 <= nums1.length <= nums2.length <= 1000
	0 <= nums1[i], nums2[i] <= 10^4
	All integers in nums1 and nums2 are unique.
	All the integers of nums1 also appear in nums2.


### 解析

根据题意，num1 是 num2 的子集，找出 num1 中每个元素在 num2 中的对应位置之后的第一个大于它的数，如果没有则结果为 -1 。我这里定义了一个方法 find ，就是为了在 num2 中的 idx 位置之后找比 nums2[idx] 大的数字。直接遍历 num1 ，得到元素在 num2 中的索引，然后使用我定义的 find 函数，得到的结果追加到 result 中，遍历结束即可得到结果。


### 解答
				

	class Solution(object):
	    def nextGreaterElement(self, nums1, nums2):
	        """
	        :type nums1: List[int]
	        :type nums2: List[int]
	        :rtype: List[int]
	        """
	        def find(A, idx):
	            result = -1
	            for i in range(idx, len(A)):
	                if A[i]>A[idx]:
	                    result = A[i]
	                    break
	            return result
	        result = []
	        for n in nums1:
	            idx = nums2.index(n)
	            result.append(find(nums2, idx))
	        return result
	        
            	      
			
### 运行结果

	Runtime: 80 ms, faster than 21.98% of Python online submissions for Next Greater Element I.
	Memory Usage: 13.3 MB, less than 100.00% of Python online submissions for Next Greater Element I.

### 解析

上面是按照题意解答的一种思路，比较简单，还有一种方法利用了字典和栈，因为每个 nums1 中的数字都在 nums2 中，遍历 nums2 中的每个数字，找出每个元素及其后面第一个大的元素存入字典 d 中，然后遍历 num1 中的每个数字，从 d 中直接获取值，如果没有则直接返回 -1 。

### 解答

	class Solution(object):
	    def nextGreaterElement(self, nums1, nums2):
	        """
	        :type nums1: List[int]
	        :type nums2: List[int]
	        :rtype: List[int]
	        """
	        result = []
	        stack = []
	        d = {}
	        for num in nums2:
	            while len(stack) and (stack[-1]<num):
	                d[stack.pop()] = num
	            stack.append(num)
	        for num in nums1:
	            result.append(d.get(num,-1))
	        return result
	        

### 运行结果

	Runtime: 28 ms, faster than 96.30% of Python online submissions for Next Greater Element I.
	Memory Usage: 13.8 MB, less than 33.55% of Python online submissions for Next Greater Element I.

原题链接：https://leetcode.com/problems/next-greater-element-i



您的支持是我最大的动力
