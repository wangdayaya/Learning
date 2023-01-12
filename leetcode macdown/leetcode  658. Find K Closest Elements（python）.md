leetcode  658. Find K Closest Elements（python）




### 描述


Given a sorted integer array arr, two integers k and x, return the k closest integers to x in the array. The result should also be sorted in ascending order. An integer a is closer to x than an integer b if:

	|a - x| < |b - x|, or
	|a - x| == |b - x| and a < b


Example 1:

	Input: arr = [1,2,3,4,5], k = 4, x = 3
	Output: [1,2,3,4]

	
Example 2:

	Input: arr = [1,2,3,4,5], k = 4, x = -1
	Output: [1,2,3,4]






Note:


	1 <= k <= arr.length
	1 <= arr.length <= 10^4
	arr is sorted in ascending order.
	-10^4 <= arr[i], x <= 10^4


### 解析

根据题意，给定一个已经按照升序排列好的数组 arr ，另外还有两个整数 k 和 x ，题目要求我们找出最靠近 x 的 k 个数字，并且要求最后返回的结果是按照升序排列的。题目还给出了什么叫“最靠近”的意思，其实根本没必要看。

这道题其实就是个 Eazy 难度的，一开始我打算把题目看懂，但是描述有点乱，最后看了几次都模模糊糊干脆放弃了，直接看例子比较清楚，题目要解决的事情其实就是个排序问题，我们使用的是 python ，直接调用内置函数 sort ，将排序的要求按照题目写清楚，得到新的排序后的 arr ，最后将返回的前 k 个结果在进行升序排序返回即可。

时间复杂度为 O(NlogN) ，空间复杂度为 O(logN) ，N 为 arr 的长度。


### 解答

	class Solution(object):
	    def findClosestElements(self, arr, k, x):
	        """
	        :type arr: List[int]
	        :type k: int
	        :type x: int
	        :rtype: List[int]
	        """
	        arr.sort(key=lambda a:abs(a-x))
	        return sorted(arr[:k])


### 运行结果

	Runtime: 591 ms, faster than 29.27% of Python online submissions for Find K Closest Elements.
	Memory Usage: 14.8 MB, less than 95.06% of Python online submissions for Find K Closest Elements.

### 解析

另外这道题还能使用删除元素法来解题，因为最后 arr 是有序的，并且最后只需要留下距离 x 最近的 k 个元素即可，那么在 arr 中离 x 最远的只有可能是 arr[0] 和 arr[-1] ，所以我们只需要通过比较之后，不断删除距离 x 较远的元素最后得到的长为 k 的数组就是结果，直接返回即可。

时间复杂度为 O(N) ，空间复杂度为 O(1) ，N 为 arr 的长度。

### 解答
	class Solution(object):
	    def findClosestElements(self, arr, k, x):
	        """
	        :type arr: List[int]
	        :type k: int
	        :type x: int
	        :rtype: List[int]
	        """
	        N = len(arr)
	        while N != k:
	            if abs(arr[-1] - x) >= abs(arr[0] - x):
	                arr.remove(arr[-1])
	            elif abs(arr[-1] - x) < abs(arr[0] - x):
	                arr.remove(arr[0])
	            N -= 1
	        return arr
	 

### 运行结果

	Runtime: 3285 ms, faster than 5.11% of Python online submissions for Find K Closest Elements.
	Memory Usage: 15.1 MB, less than 65.08% of Python online submissions for Find K Closest Elements.


### 原题链接
https://leetcode.com/problems/find-k-closest-elements/submissions/



您的支持是我最大的动力
