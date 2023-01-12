leetcode  905. Sort Array By Parity（python）

### 描述

Given an integer array nums, move all the even integers at the beginning of the array followed by all the odd integers.

Return any array that satisfies this condition.





Example 1:

	Input: nums = [3,1,2,4]
	Output: [2,4,3,1]
	Explanation: The outputs [4,2,3,1], [2,4,1,3], and [4,2,1,3] would also be accepted.


	
Example 2:

	Input: nums = [0]
	Output: [0]




Note:

	1 <= nums.length <= 5000
	0 <= nums[i] <= 5000


### 解析


根据题意，就是给出了一个整数列表 nums ，要求将其中的全部偶数都放到全部的奇数前面。思路比较简单，直接在 nums 原列表上进行操作：

* 用 index 记录处理元素的索引，用 count 记录处理了多少个数字
* 如果 nums[index] 是奇数，就将其放到 nums 的末尾，同时 count 加一
* 如果 nums[index] 是偶数，则不发生变化，将 count 和 index 同时加一
* 遍历结束，返回 nums 即可

### 解答
				

	class Solution(object):
	    def sortArrayByParity(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: List[int]
	        """
	        L = len(nums)
	        index = 0
	        count = 0
	        while count<L:
	            if nums[index]%2==1:
	                nums.append(nums.pop(index))
	            else:
	                index += 1
	            count += 1
	        return nums
            	      
			
### 运行结果

	Runtime: 72 ms, faster than 20.73% of Python online submissions for Sort Array By Parity.
	Memory Usage: 14.3 MB, less than 74.48% of Python online submissions for Sort Array By Parity.

### 解析

另外可以直接在原列表中进行操作，这样不用占用额外的空间。因为偶数要放在前面，奇数放在后面，所以：

* 初始化最左边索引 l 和最右边索引 r
* 当 l<r 的时候，一直执行 while ，将前面的奇数和后面的偶数进行交换，并不断更新 l 和 r 的值
* 遍历结束得到的新的 nums 即为答案

### 解答

	class Solution(object):
	    def sortArrayByParity(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: List[int]
	        """
	        l = 0
	        r = len(nums)-1
	        while l<r:
	            if nums[l]%2==1 and nums[r]%2==0:
	                nums[l], nums[r] = nums[r], nums[l]
	            if nums[l]%2==0: l+=1
	            if nums[r]%2==1: r-=1
	        return nums
	


### 运行结果


	Runtime: 68 ms, faster than 41.56% of Python online submissions for Sort Array By Parity.
	Memory Usage: 14.5 MB, less than 16.04% of Python online submissions for Sort Array By Parity.

### 解析

还可以使用 python 的内置函数 sort ，自定义比较器即可。
### 解答

	class Solution(object):
	    def sortArrayByParity(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: List[int]
	        """
	        nums.sort(key = lambda x:x%2)
	        return nums
	

### 运行结果

	Runtime: 65 ms, faster than 41.98% of Python online submissions for Sort Array By Parity.
	Memory Usage: 14.2 MB, less than 93.12% of Python online submissions for Sort Array By Parity.

原题链接：https://leetcode.com/problems/sort-array-by-parity/



您的支持是我最大的动力
