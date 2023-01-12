leetcode 11. Container With Most Water （python）




### 描述


You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the i<sub>th</sub> line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.Notice that you may not slant the container.


Example 1:


![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/17/question_11.jpg)

	Input: height = [1,8,6,2,5,4,8,3,7]
	Output: 49
	Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
	



Note:

	n == height.length
	2 <= n <= 10^5
	0 <= height[i] <= 10^4


### 解析

根据题意，给定一个长度为 n 的整数数组 height 。 绘制了 n 条垂直线，使得第 i 条线的低点和高点分别是是 (i, 0) 和 (i, height[i])。

找到两条线，它们与 x 轴一起形成一个容器，使得容器中的水最多。返回容器可以存储的最大水量。

这道题很有意思，因为很贴近生活场景，我们有时候就会碰到这种在某种条件下求极值的情况。其实题目考察的既有贪心算法，又有双指针，结合两个知识点的考题很少见。

如果是暴力的解法肯定会超时，所以用双指针就可以 AC 。我们定义 S(i, j) 是在 i 到 j 的可容纳面积，无论是最左边的柱子右移还是最右边的柱子左移，都会导致宽度变小，所以我们想要保证容量最大，只能在柱子的高度上做文章，有两种操作：

* 移开较短的柱子，这样容器的容量由较短的那根决定， min(h[i], h[j]) * (j-i) 可能会容纳更多的水，当然如果碰到更矮的柱子这样也可能会容纳更少的水
* 移开较长的柱子，容器的容量由较短的那根决定，所以容器的较短柱子 min(h[i], h[j]) 将可能保持不变或更短，因此容量肯定会变小

因此，我们可以使用两个指针指向容器的左右柱子。 我们在每一轮中移开较短的柱子，更新最大容量，直到两个指针相遇，最后返回最大面积即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。



### 解答
				

	class Solution(object):
	    def maxArea(self, height):
	        """
	        :type height: List[int]
	        :rtype: int
	        """
	        if len(height) == 2: return min(height[0], height[1]) * 1
	        N = len(height)
	        left = 0
	        right = N - 1
	        result =  min(height[left], height[right]) * (right - left)
	        while left < right:
	            if height[left] <= height[right]:
	                left += 1
	            else:
	                right -= 1
	            result = max( result, min( height[right], height[left]) * (right-left) )
	        return result
            	      
			
### 运行结果

	Runtime: 763 ms, faster than 43.95% of Python online submissions for Container With Most Water.
	Memory Usage: 23.9 MB, less than 53.52% of Python online submissions for Container With Most Water.


### 原题链接


https://leetcode.com/problems/container-with-most-water/


您的支持是我最大的动力
