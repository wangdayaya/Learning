leetcode 2074. Reverse Nodes in Even Length Groups （python）

### 描述

You are given the head of a linked list.

The nodes in the linked list are sequentially assigned to non-empty groups whose lengths form the sequence of the natural numbers (1, 2, 3, 4, ...). The length of a group is the number of nodes assigned to it. In other words,

* The 1st node is assigned to the first group.
* The 2nd and the 3rd nodes are assigned to the second group.
* The 4th, 5th, and 6th nodes are assigned to the third group, and so on.

Note that the length of the last group may be less than or equal to 1 + the length of the second to last group.

Reverse the nodes in each group with an even length, and return the head of the modified linked list.



Example 1:

![](https://assets.leetcode.com/uploads/2021/10/25/eg1.png)


	Input: head = [5,2,6,3,9,1,7,3,8,4]
	Output: [5,6,2,3,9,1,4,8,3,7]
	Explanation:
	- The length of the first group is 1, which is odd, hence no reversal occurrs.
	- The length of the second group is 2, which is even, hence the nodes are reversed.
	- The length of the third group is 3, which is odd, hence no reversal occurrs.
	- The length of the last group is 4, which is even, hence the nodes are reversed.

	
Example 2:
![](https://assets.leetcode.com/uploads/2021/10/25/eg2.png)

	Input: head = [1,1,0,6]
	Output: [1,0,1,6]
	Explanation:
	- The length of the first group is 1. No reversal occurrs.
	- The length of the second group is 2. The nodes are reversed.
	- The length of the last group is 1. No reversal occurrs.



Example 3:

![](https://assets.leetcode.com/uploads/2021/11/17/ex3.png)

	Input: head = [1,1,0,6,5]
	Output: [1,0,1,5,6]
	Explanation:
	- The length of the first group is 1. No reversal occurrs.
	- The length of the second group is 2. The nodes are reversed.
	- The length of the last group is 2. The nodes are reversed.

	
Example 4:

![](https://assets.leetcode.com/uploads/2021/10/28/eg3.png)

	Input: head = [2,1]
	Output: [2,1]
	Explanation:
	- The length of the first group is 1. No reversal occurrs.
	- The length of the last group is 1. No reversal occurrs.

	
Example 5:

	Input: head = [8]
	Output: [8]
	Explanation: There is only one group whose length is 1. No reversal occurrs.


Note:

The number of nodes in the list is in the range [1, 10^5].
0 <= Node.val <= 10^5

### 解析


根据题意，给定获得链表的头指针 head 。将链表从左到右分成一个节点为一组，两个节点为一组，三个节点为一组，以此类推，但是最后一组的长度不确定，可能小于正常的分组长度。将每组中长度为偶数的节点反转，返回修改后的链表的头部。

读完题之后，题意是比较明显的。最简单的方法莫过于将链表中的值都存入一个列表中，然后按照题意进行变化重新连接起来一个新的链表即可，但是这种方法太取巧了，比赛的时候可以投机取巧，平时练习题还是扎实一点比较好，题目要考察的肯定是链表的遍历、拼接和反转，那我就按照题目考察的内容，先将链表按照 1 、 2、 3 ...  的长度进行分组，然后对偶数长度的部分进行链表的局部反转，最后将左右的分组进行拼接即可。

### 解答
				
	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def reverseEvenLengthGroups(self, head):
	        """
	        :type head: Optional[ListNode]
	        :rtype: Optional[ListNode]
	        """
	        if not head or not head.next: return head
	        heads = []
	        lens = []
	        gLength = 1
	        while True:
	            heads.append(head)
	            count = 1
	            for _ in range(gLength-1):
	                if not head.next: break
	                head = head.next
	                count += 1
	            lens.append(count)
	            if not head.next: break
	            nxt = head.next
	            head.next = None
	            head = nxt
	            gLength += 1
	            
	        for i in range(len(lens)):
	            if lens[i]%2==0:
	                heads[i] = self.reverse(heads[i])
	                
	        for i in range(len(heads)-1):
	            t = heads[i]
	            while t.next:
	                t = t.next
	            t.next = heads[i+1]
	        
	        return heads[0]
	    
	    def reverse(self, head):
	        cur = head
	        last = None
	        while cur:
	            nxt = cur.next
	            cur.next = last
	            last = cur
	            cur = nxt
	        return last            

            	      
			
### 运行结果
	
	Runtime: 3092 ms, faster than 71.33% of Python online submissions for Reverse Nodes in Even Length Groups.
	Memory Usage: 91.9 MB, less than 37.06% of Python online submissions for Reverse Nodes in Even Length Groups.



原题链接：https://leetcode.com/problems/reverse-nodes-in-even-length-groups/



您的支持是我最大的动力
