leetcode  421. Maximum XOR of Two Numbers in an Array（python）




### 描述

Given an integer array nums, return the maximum result of nums[i] XOR nums[j], where 0 <= i <= j < n.





Example 1:

	Input: nums = [3,10,5,25,2,8]
	Output: 28
	Explanation: The maximum result is 5 XOR 25 = 28.

	
Example 2:

	Input: nums = [14,70,53,83,49,91,36,80,92,51,66,70]
	Output: 127





Note:


	1 <= nums.length <= 2 * 10^5
	0 <= nums[i] <= 2^31 - 1

### 解析

根据题意，给出了一个整数数组 nums ，在 0 <= i <= j < n 范围内，返回 nums[i] XOR nums[j] 最大的结果。结合限制条件我们一眼就能发现这道题的时间复杂度必须在 O(n) 左右的，因为 nums 的长度的最大长度为 2 * 10^5 ，而且每个数字还挺大，在计算量上会很大，如果用传统的暴力解法，两层循环去找最大值，时间复杂度为 O(n^2) ，肯定是会超时的。所以这道题的难点在于要找一种算法，将 O(n^2) 的时间复杂度压缩到 O(n) 的时间复杂度。

乍一看没什么思路，因为常规的思路怎么想都是 O(n^2) ，看了大佬的解法，深受启发，其实还是有规律可循的，因为我们假如 nums 中有一个数字 010101  ，我们想找到一个数字与其异或运算让其理想状态最大，那么肯定是对应二进制位相反组成的数字 101010 ，因为只有这样异或运算的结果才是最大的，但是 101010 不一定出现在 nums ，所以我们要用类似贪心的思想：

* 从剩余的 nums 中的元素中，找二进制第一个位为 1 的数字集合
* 如果集合中有多个元素出现，再找第二位是 0 的数字集合
* 当然如果可选的元素某一位上都没有合适的，那么也都保留在集合中，因为没有更好的选择了，再继续找后面的位即可
* 我们重复上面的过程，直到找到的集合中只有一个元素，那么它就是在 nums 中与 010101 异或能得到最大结果的数字

其实介绍到这里，我们应该就知道，要看前 n 个位的数字是否有效，最快的方法就是用到 Trie 树结构，因为将这么多的数字都变成一个 n 位的二进制存放到同一个树里面，我们才可以轻松按照上面的搜索思路找到最大的异或结果，时间复杂度最多也就是 O(32) ，因为我们不需要遍历 nums ，只需要找前 n（n<=32） 层 Trie 树节点即可。另外一个就是 Trie 可以节省很多的空间。

### 解答
				

	class TrieNode:
	    def __init__(self):
	        self.children = {}                      
	        self.val = 0                                  
	class Trie:
	    def __init__(self, n):
	        self.root = TrieNode()                        
	        self.n = n                                    
	        
	    def add_num(self, num):
	        node = self.root 
	        for tmp in range(self.n, -1, -1):           
	            val = 1 if num & (1 << tmp) else 0     
	            if val not in node.children:
	                node.children[val] = TrieNode()
	            node = node.children[val]
	        node.val = num
	class Solution(object):
	    def findMaximumXOR(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        L = len(bin(max(nums))) - 2             
	        trie = Trie(L)
	        for num in nums: 
	            trie.add_num(num)            
	        result = 0
	        for num in nums:                              
	            node = trie.root 
	            for tmp in range(L, -1, -1):
	                val = 1 if num & (1 << tmp) else 0  
	                node = node.children[1-val] if 1-val in node.children else node.children[val] 
	            result = max(result, num ^ node.val)          
	        return result
			
### 运行结果

	Runtime: 6344 ms, faster than 13.23% of Python3 online submissions for Maximum XOR of Two Numbers in an Array.
	Memory Usage: 154.7 MB, less than 5.19% of Python3 online submissions for Maximum XOR of Two Numbers in an Array.



### 原题链接

https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/

您的支持是我最大的动力
