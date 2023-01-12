leetcode  835. Image Overlap（python）




### 描述

You are given two images, img1 and img2, represented as binary, square matrices of size n x n. A binary matrix has only 0s and 1s as values. We translate one image however we choose by sliding all the 1 bits left, right, up, and/or down any number of units. We then place it on top of the other image. We can then calculate the overlap by counting the number of positions that have a 1 in both images. Note also that a translation does not include any kind of rotation. Any 1 bits that are translated outside of the matrix borders are erased. Return the largest possible overlap.



Example 1:

![](https://assets.leetcode.com/uploads/2020/09/09/overlap1.jpg)

	Input: img1 = [[1,1,0],[0,1,0],[0,1,0]], img2 = [[0,0,0],[0,1,1],[0,0,1]]
	Output: 3
	Explanation: We translate img1 to right by 1 unit and down by 1 unit.

![](https://assets.leetcode.com/uploads/2020/09/09/overlap_step1.jpg)

	The number of positions that have a 1 in both images is 3 (shown in red).

![](https://assets.leetcode.com/uploads/2020/09/09/overlap_step2.jpg)

	
Example 2:

	Input: img1 = [[1]], img2 = [[1]]
	Output: 1


Example 3:

	Input: img1 = [[0]], img2 = [[0]]
	Output: 0



Note:

	n == img1.length == img1[i].length
	n == img2.length == img2[i].length
	1 <= n <= 30
	img1[i][j] is either 0 or 1.
	img2[i][j] is either 0 or 1.


### 解析

根据题意，给定两个图像 img1 和 img2 ，表示为大小为 n x n 的二进制正方形矩阵。二进制矩阵中只有 0 和 1 作为值。我们将其中一张图片的所有的 1 通过向左，向右，向上，向下滑动任意数量的单位来转换一个图像。然后，我们将其放在另一个图像的上，然后我们可以通过计算两个图像中都具有 1 的位置数来计算重叠。需要注意的是平移不包括任何类型的旋转。任何在移动到矩阵边框之外的 1 位都将被清除。返回最大可能的重叠。

因为题目中给出的矩阵长度最大为 30 ，如果用暴力的方法，虽然不一定会超时，但是代码肯定不好写，而且经过了很多无用的步骤，我们通过观察可以发现，当将两个矩阵重叠的时候，如果对每个矩阵计算其元素为 1 的位置索引的 hash 值矩阵，通过计数器对这两个新矩阵中的 hash 值进行两两相减运算，然后通过计数器计数能找到最大的重合数。

所以我们先通过上面的思路得到 img1 和 img2 对应的经过 hash 之后的矩阵 L_img1 和 L_img2 ，然后我们将两个矩阵中的值经过两两相减运算，然后通过计数器 c 可以得到出现次数最多的值，也就是最大的重合数。

时间复杂度为 O(N^2) ，空间复杂度为 O(N^2) 。

### 解答

	class Solution(object):
	    def largestOverlap(self, img1, img2):
	        """
	        :type img1: List[List[int]]
	        :type img2: List[List[int]]
	        :rtype: int
	        """
	        N = len(img1)
	        L_img1 = [i / N * 100 + i % N for i in range(N * N) if img1[i / N][i % N]]
	        L_img2 = [i / N * 100 + i % N for i in range(N * N) if img2[i / N][i % N]]
	        c = collections.Counter(i - j for i in L_img1 for j in L_img2)
	        return max(c.values() or [0])

### 运行结果

	Runtime: 343 ms, faster than 94.44% of Python online submissions for Image Overlap.
	Memory Usage: 13.9 MB, less than 55.56% of Python online submissions for Image Overlap.
	
### 原题链接

https://leetcode.com/problems/image-overlap/


您的支持是我最大的动力
