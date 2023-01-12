leetcode  406. Queue Reconstruction by Height（python）

### 描述


You are given an array of people, people, which are the attributes of some people in a queue (not necessarily in order). Each people[i] = [h<sub>i</sub>, k<sub>i</sub>] represents the i<sub>th</sub> person of height h<sub>i</sub> with exactly k<sub>i</sub> other people in front who have a height greater than or equal to h<sub>i</sub>.

Reconstruct and return the queue that is represented by the input array people. The returned queue should be formatted as an array queue, where queue[j] = [h<sub>j</sub>, k<sub>j</sub>] is the attributes of the jth person in the queue (queue[0] is the person at the front of the queue).


Example 1:


	Input: people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
	Output: [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
	Explanation:
	Person 0 has height 5 with no other people taller or the same height in front.
	Person 1 has height 7 with no other people taller or the same height in front.
	Person 2 has height 5 with two persons taller or the same height in front, which is person 0 and 1.
	Person 3 has height 6 with one person taller or the same height in front, which is person 1.
	Person 4 has height 4 with four people taller or the same height in front, which are people 0, 1, 2, and 3.
	Person 5 has height 7 with one person taller or the same height in front, which is person 1.
	Hence [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] is the reconstructed queue.
	
Example 2:


	Input: people = [[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]]
	Output: [[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]




Note:

* 	1 <= people.length <= 2000
* 	0 <= h<sub>i</sub> <= 10^6
* 	0 <= k<sub>i</sub> < people.length
* 	It is guaranteed that the queue can be reconstructed.

### 解析

根据题意，给定一组人 people ，这是队列中某些人的属性。 每个 people[i] = [h<sub>i</sub>, k<sub>i</sub>] 代表第 i 个身高为 h<sub>i</sub> 的人，前面正好有 k<sub>i</sub> 其他身高大于或等于 h<sub>i</sub> 的人。题目要求重构并返回输入数组 people 所代表的队列。 返回的队列应格式化为新的数组 queue ，其中 queue[j] = [h<sub>j</sub>, k<sub>j</sub>] 为队列中第 j 个人的属性（ queue[0] 为队列最前面的人）。

题目很简单，就是让我们把 people 中的元素重新排列，满足每个元素的身高和位置属性，并将排列后的数组返回。因为要满足前面身高大于等于自己的人数要求，所以先将身高 h 降序排列，同时按 k 升序排列，再按照 k 考虑我们插入的位置，如对例子一进行变化：
	
	 [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]] --> [[7,0], [7,1], [6,1], [5,0], [5,2], [4,4]]
	
然后定义一个空列表 result ，开始遍历上述的列表进行插入：

	[7,0] 插入到索引为 0 的位置: [[7,0]]
	[7,1] 插入到索引为 1 的位置，前面正好有 1 个身高大于等于他的人: [[7,0], [7,1]]
	[6,1] 插入到索引为 1 的位置，前面正好有 1 个身高大于等于他的人: [[7, 0], [6, 1], [7, 1]] 
	[5,0] 插入到索引为 0 的位置，前面正好有 0 个身高大于等于他的人: [[5, 0], [7, 0], [6, 1], [7, 1]] 
	[5,2] 插入到索引为 2 的位置，前面正好有 2 个身高大于等于他的人: [[5, 0], [7, 0], [5, 2], [6, 1], [7, 1]] 
	[4,4]插入到索引为 4 的位置，前面正好有 4 个身高大于等于他的人: [[5, 0], [7, 0], [5, 2], [6, 1], [4, 4], [7, 1]]
	

### 解答
				

	class Solution(object):
	    def reconstructQueue(self, people):
	        """
	        :type people: List[List[int]]
	        :rtype: List[List[int]]
	        """
	        people.sort(reverse=True)
	        result = []
	        for h,k in sorted(people, key=lambda x: (-x[0], x[1])):
	            result.insert(k, [h,k])
	        return result
            	      
			
### 运行结果

	Runtime: 80 ms, faster than 72.31% of Python online submissions for Queue Reconstruction by Height.
	Memory Usage: 14.1 MB, less than 45.38% of Python online submissions for Queue Reconstruction by Height.


原题链接：https://leetcode.com/problems/queue-reconstruction-by-height/



您的支持是我最大的动力
