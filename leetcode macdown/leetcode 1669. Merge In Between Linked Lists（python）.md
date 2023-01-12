leetcode  1669. Merge In Between Linked Lists（python）

### 描述


You are given two linked lists: list1 and list2 of sizes n and m respectively.

Remove list1's nodes from the a<sup>th</sup> node to the b<sup>th</sup> node, and put list2 in their place.

The blue edges and nodes in the following figure incidate the result:

![](https://assets.leetcode.com/uploads/2020/11/05/fig1.png)

Build the result list and return its head.


Example 1:

![](https://assets.leetcode.com/uploads/2020/11/05/merge_linked_list_ex1.png)

	Input: list1 = [0,1,2,3,4,5], a = 3, b = 4, list2 = [1000000,1000001,1000002]
	Output: [0,1,2,1000000,1000001,1000002,5]
	Explanation: We remove the nodes 3 and 4 and put the entire list2 in their place. The blue edges and nodes in the above figure indicate the result.

	
Example 2:

![](https://assets.leetcode.com/uploads/2020/11/05/merge_linked_list_ex2.png)

	Input: list1 = [0,1,2,3,4,5,6], a = 2, b = 5, list2 = [1000000,1000001,1000002,1000003,1000004]
	Output: [0,1,1000000,1000001,1000002,1000003,1000004,6]
	Explanation: The blue edges and nodes in the above figure indicate the result.





Note:

	3 <= list1.length <= 10^4
	1 <= a <= b < list1.length - 1
	1 <= list2.length <= 10^4


### 解析

根据题意，就是给出了两个长度分别为 n 和 m 的链表 list1 和 list2 ，同时又给出了两个数字 a 和 b ，题目要求我们移除 list1 中从 a<sup>th</sup> 节点到 b<sup>th</sup> 节点，并且将 list2 接入到移除的节点的位置。思路比较简单：

* 先初始化变量 L1\_H 为 list1 表示一会要将 list2 接入 list1 中的头节点，初始化变量 L1_tail 为 list1 表示在 list2 之后要接上的 list1 的剩余尾部节点。
* 使用 while 循环不断往后寻找之后将 list2 接入 list1 的头节点 L1_H
* 使用 while 循环不断往后寻找 list1 的尾部剩余节点 L1_tail
* 将 list2 接到 L1_H 之后
* 使用 while 循环不断往后寻找 list2 的尾节点
* 将 L1_tail 接入到 list2 的尾节点之后
* 返回 list1 即是最新的拼接好的链表


### 解答
				

	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def mergeInBetween(self, list1, a, b, list2):
	        """
	        :type list1: ListNode
	        :type a: int
	        :type b: int
	        :type list2: ListNode
	        :rtype: ListNode
	        """
	        L1_H = list1
	        L1_tail = list1
	        while a-1>0:
	            L1_H = L1_H.next
	            a -= 1
	        while b-1>=0:
	            L1_tail = L1_tail.next
	            b -= 1
	        L1_H.next = list2
	        while list2:
	            if list2.next:
	                list2 = list2.next
	            else:
	                break
	        list2.next = L1_tail.next
	        return list1
            	      
			
### 运行结果

	Runtime: 676 ms, faster than 16.10% of Python online submissions for Merge In Between Linked Lists.
	Memory Usage: 24.5 MB, less than 61.86% of Python online submissions for Merge In Between Linked Lists.


### 解析

还可以对上述的过程简化一下，思路和上面一样。其实这道题按照正常的逻辑思维把代码写出来就肯定是对的，只不过高手可能对代码的组织更加紧凑合理，我有点不自信去论坛看其他大神的解法，发现其实基本上都是大同小异的对两个链表的遍历和拼接操作。我觉得可能对不起这个 Medium 的难度。


### 解答


	class ListNode(object):
	    def __init__(self, val=0, next=None):
	        self.val = val
	        self.next = next
	class Solution(object):
	    def mergeInBetween(self, list1, a, b, list2):
	        """
	        :type list1: ListNode
	        :type a: int
	        :type b: int
	        :type list2: ListNode
	        :rtype: ListNode
	        """
	        head = list1
	        for _ in range(a-1):
	            head = head.next
	        cur = head.next
	        for _ in range(b-a):
	            cur = cur.next
	        head.next = list2
	        while head.next:
	            head = head.next
	        if cur.next:
	            head.next = cur.next
	        return list1
### 运行结果	        
	        
	Runtime: 469 ms, faster than 93.22% of Python online submissions for Merge In Between Linked Lists.
	Memory Usage: 24.8 MB, less than 11.02% of Python online submissions for Merge In Between Linked Lists.  

原题链接：https://leetcode.com/problems/merge-in-between-linked-lists/



您的支持是我最大的动力
