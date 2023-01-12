leetcode  2023. Number of Pairs of Strings With Concatenation Equal to Target（python）

### 描述

Given an array of digit strings nums and a digit string target, return the number of pairs of indices (i, j) (where i != j) such that the concatenation of nums[i] + nums[j] equals target.





Example 1:

	Input: nums = ["777","7","77","77"], target = "7777"
	Output: 4
	Explanation: Valid pairs are:
	- (0, 1): "777" + "7"
	- (1, 0): "7" + "777"
	- (2, 3): "77" + "77"
	- (3, 2): "77" + "77"

	
Example 2:

	Input: nums = ["123","4","12","34"], target = "1234"
	Output: 2
	Explanation: Valid pairs are:
	- (0, 1): "123" + "4"
	- (2, 3): "12" + "34"


Example 3:

	Input: nums = ["1","1","1"], target = "11"
	Output: 6
	Explanation: Valid pairs are:
	- (0, 1): "1" + "1"
	- (1, 0): "1" + "1"
	- (0, 2): "1" + "1"
	- (2, 0): "1" + "1"
	- (1, 2): "1" + "1"
	- (2, 1): "1" + "1"



Note:

	
	2 <= nums.length <= 100
	1 <= nums[i].length <= 100
	2 <= target.length <= 100
	nums[i] and target consist of digits.
	nums[i] and target do not have leading zeros.

### 解析

根据题意，就是给出了一个数字字符串列表 nums 和一个数字字符串 target ，如果nums 中的索引 (i，j) 且 i != j 对应的两个数字字符串进行拼接之后 nums[i]+nums[j] 等于 target ，就将计数器 result 加一，最后返回 result 值即可。

这道题其实就是在考察排列组合，但是也可以暴力解法，那既然这样我肯定是用后者才符合我暴力美学的身份，思路比较简单：

* 初始化计数器 result 为 0 ，nums 的长度 N ，target 的长度为 M
* 第一层对 range(N-1) 中的每个元素 i 进行循环，第二层对 range(i+1,N) 中的每个元素 j 进行循环
* 如果 nums[i] 和 nums[j] 的长度总和等于 M ，那么就继续判断如果 nums[i] 和 nums[j] 的字符串拼接等于 target ，计数器 result 加一，如果 nums[j] 和 nums[i] 的字符串拼接等于 target ，计数器 result 加一
* 循环结束，直接返回 result 即可


### 解答
				
	class Solution(object):
	    def numOfPairs(self, nums, target):
	        """
	        :type nums: List[str]
	        :type target: str
	        :rtype: int
	        """
	        result = 0
	        N = len(nums)
	        M = len(target)
	        for i in range(N-1):
	            for j in range(i+1,N):
	                if len(nums[i]) + len(nums[j]) == M:
	                    if nums[i] + nums[j] == target:
	                        result+=1
	                    if nums[j] + nums[i] == target:
	                        result+=1
	        return result

            	      
			
### 运行结果

	Runtime: 51 ms, faster than 66.99% of Python online submissions for Number of Pairs of Strings With Concatenation Equal to Target.
	Memory Usage: 13.6 MB, less than 51.67% of Python online submissions for Number of Pairs of Strings With Concatenation Equal to Target.

### 解析

上面的思路其实就是暴力循环求解，没有体现出这个题考察的排列组合的内容，我们还可以使用 python 的内置函数：

* 使用 itertools.permutations 将 nums 中的字符串进行两两组合
* 遍历得到的字符串组合列表，将每个字符串组合进行拼接判断是否等于 target ，如果等于则计数器加一
* 遍历结束直接返回结果 result 

总体来说这个题比较简单，就是个 easy 级别的吧，对不起它 Medium 的标志。

### 解答

	class Solution(object):
	    def numOfPairs(self, nums, target):
	        """
	        :type nums: List[str]
	        :type target: str
	        :rtype: int
	        """
	        result = 0
	        pairs = [(i,j) for i, j in permutations(nums, 2)]
	        for p in pairs:
	            if p[0]+p[1] == target:
	                result += 1
	        return result

### 运行结果

	Runtime: 183 ms, faster than 5.26% of Python online submissions for Number of Pairs of Strings With Concatenation Equal to Target.
	Memory Usage: 14.1 MB, less than 11.96% of Python online submissions for Number of Pairs of Strings With Concatenation Equal to Target.
	
原题链接：https://leetcode.com/problems/number-of-pairs-of-strings-with-concatenation-equal-to-target/



您的支持是我最大的动力
