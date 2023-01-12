Offer 驾到，掘友接招！我正在参与2022春招系列活动-刷题打卡任务，点击查看[活动详情](https://juejin.cn/post/7069661622012215309/ "https://juejin.cn/post/7069661622012215309/")


### 描述

We stack glasses in a pyramid, where the first row has 1 glass, the second row has 2 glasses, and so on until the 100<sup>th</sup> row.  Each glass holds one cup of champagne.

Then, some champagne is poured into the first glass at the top.  When the topmost glass is full, any excess liquid poured will fall equally to the glass immediately to the left and right of it.  When those glasses become full, any excess champagne will fall equally to the left and right of those glasses, and so on.  (A glass at the bottom row has its excess champagne fall on the floor.)

For example, after one cup of champagne is poured, the top most glass is full.  After two cups of champagne are poured, the two glasses on the second row are half full.  After three cups of champagne are poured, those two cups become full - there are 3 full glasses total now.  After four cups of champagne are poured, the third row has the middle glass half full, and the two outside glasses are a quarter full, as pictured below.

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4a9357e9109e436e8303a96e58abbfa2~tplv-k3u1fbpfcp-zoom-1.image)

Now after pouring some non-negative integer cups of champagne, return how full the j<sup>th</sup> glass in the i<sup>th</sup> row is (both i and j are 0-indexed.)

Example 1:


	Input: poured = 1, query_row = 1, query_glass = 1
	Output: 0.00000
	Explanation: We poured 1 cup of champange to the top glass of the tower (which is indexed as (0, 0)). There will be no excess liquid so all the glasses under the top glass will remain empty.
	
Example 2:

	Input: poured = 2, query_row = 1, query_glass = 1
	Output: 0.50000
	Explanation: We poured 2 cups of champange to the top glass of the tower (which is indexed as (0, 0)). There is one cup of excess liquid. The glass indexed as (1, 0) and the glass indexed as (1, 1) will share the excess liquid equally, and each will get half cup of champange.


Example 3:


	Input: poured = 100000009, query_row = 33, query_glass = 17
	Output: 1.00000
	



Note:

	0 <= poured <= 10^9
	0 <= query_glass <= query_row < 100


### 解析

根据题意，就是把杯子从上到下摆成一个金字塔形状，第一层有一个，第二层有两个，以此类推，然后从最上面倒酒，如果上面的杯子装满就会溢出均匀流到下面的杯子里，最下面一层如果溢出则流到地上。现在在倒了正整数杯香槟之后，返回第 i 行中第 j 个杯子的装满程度（i 和 j 都是 0 开始的索引）。

这个题很有意思，很贴近生活，尽管题目说的很啰嗦，但是我们直接看图结合例子就能明白题意是什么意思，还是比较有难度的，因为提交错了一次我才发现，原来同样一行中每个杯子被装满的速度是不一样的，同一行两边的杯子比中间的杯子装满需要的时间要长的多，这就会导致同一行的杯子有的已经装满溢出到下面的了，有的还不满，平时没有仔细观察过生活中的现象，导致做题吃亏了，哎。

我看了大佬的解答才豁然开朗，这道题其实就是考察动态规划，我们无法去很细节的算出在倒入 poured 杯酒之后的存量，当然如果要算也是可以的只是太繁琐了，原因就是上面提到的那样。与其算每个杯子杯子现存的酒容量，不如去跟踪流过某个杯子的酒的总量。

这样我们初始化一个金字塔形状的数组 A ，因为一共倒了 poured 杯酒，肯定在顶端的 A[0][0] 地方流过了 poured ，第二行的第一个杯子流过的肯定是 (poured-1)//2 ，第二个杯子流过的也肯定是 (poured-1)//2  ，以此类推，使用两层循环，第一层循环遍历可能的行数，第二层遍历确定每个位置流过的酒的量，经过所有动态规划的计算，最后返回 min(1, A[query_row][query_glass]) 即可。


### 解答
				
	class Solution(object):
	    def champagneTower(self, poured, query_row, query_glass):
	        """
	        :type poured: int
	        :type query_row: int
	        :type query_glass: int
	        :rtype: float
	        """
	        A = [[0] * k for k in range(1, 102)]
	        A[0][0] = poured
	        for r in range(query_row + 1):
	            for c in range(r+1):
	                q = (A[r][c] - 1.0) / 2.0
	                if q > 0:
	                    A[r+1][c] += q
	                    A[r+1][c+1] += q
	
	        return min(1, A[query_row][query_glass])

            	      
			
### 运行结果

	Runtime: 132 ms, faster than 52.94% of Python online submissions for Champagne Tower.
	Memory Usage: 13.4 MB, less than 91.18% of Python online submissions for Champagne Tower.


### 原题链接


https://leetcode.com/problems/champagne-tower/


您的支持是我最大的动力
