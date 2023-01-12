leetcode  1822. Sign of the Product of an Array（python）

### 描述


There is a function signFunc(x) that returns:

* 1 if x is positive.
* -1 if x is negative.
* 0 if x is equal to 0.

You are given an integer array nums. Let product be the product of all values in the array nums.

Return signFunc(product).


Example 1:

	
	Input: nums = [-1,-2,-3,-4,3,2,1]
	Output: 1
	Explanation: The product of all values in the array is 144, and signFunc(144) = 1
	
Example 2:

	Input: nums = [1,5,0,2,-3]
	Output: 0
	Explanation: The product of all values in the array is 0, and signFunc(0) = 0


Example 3:

	Input: nums = [-1,1,-1,1,-1]
	Output: -1
	Explanation: The product of all values in the array is -1, and signFunc(-1) = -1



Note:

	1 <= nums.length <= 1000
	-100 <= nums[i] <= 100


### 解析

根据题意，就是将 nums 中的元素进行累乘之后，判断是大于 0 ，还是小于 0 ，还是等于 0 。


### 解答
				

	class Solution(object):
	    def arraySign(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        product = reduce(lambda x,y:x*y,nums)
	        if product>0:
	            return 1
	        elif product<0:
	            return -1
	        return 0
	        
            	      
			
### 运行结果

	Runtime: 40 ms, faster than 91.82% of Python online submissions for Sign of the Product of an Array.
	Memory Usage: 13.7 MB, less than 23.05% of Python online submissions for Sign of the Product of an Array.


### 解析

				
上面的思路按照题意来写代码，逻辑比较简单，另外我们可以换种思路，反正最后的结果就是判断乘积是不是大于 0 或者小于 0 或者等于零，按照乘法的规则，假如我们有一个任意的数字 n ，乘一个负数 -m ，结果就是 -n*m ，也就是绝对值会增加，但是会变成相反数，这样的话，只需要判断最后结果的符号即可。初始化结果 result 为 1 ，遍历 nums 中的每个元素，如果它小于零。则结果 result 变成相反 -1 ，如果元素为 0 则，直接返回 0，如果元素为正数则保持不变，遍历结束 result 即为最后的结果。

### 解答

	class Solution(object):
	    def arraySign(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = 1
	        for num in nums:
	            if num == 0:
	                return 0
	            if num < 0:
	                result = -result
	        return result
            	      
			
### 运行结果

	Runtime: 40 ms, faster than 89.67% of Python online submissions for Sign of the Product of an Array.
	Memory Usage: 13.4 MB, less than 95.70% of Python online submissions for Sign of the Product of an Array.

### 解析

另外，我们可以用内置函数 reduce ，直接进行快速的累乘，然后判断结果的正负或者为 0 ，这种思路和第一种一样，只不过用内置函数比较方便。

### 解答

	class Solution(object):
	    def arraySign(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        product = reduce(lambda x,y: x*y, nums)
	        return 0 if not product else 1 if product > 0 else -1

			
### 运行结果

	
	Runtime: 48 ms, faster than 40.96% of Python online submissions for Sign of the Product of an Array.
	Memory Usage: 13.6 MB, less than 48.88% of Python online submissions for Sign of the Product of an Array.


原题链接：https://leetcode.com/problems/sign-of-the-product-of-an-array/



您的支持是我最大的动力
