leetcode 2. Add Two Numbers （python）




### 描述


You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.




Example 1:

![](https://assets.leetcode.com/uploads/2020/10/02/addtwonumber1.jpg)

	Input: l1 = [2,4,3], l2 = [5,6,4]
	Output: [7,0,8]
	Explanation: 342 + 465 = 807.

	





Note:

	The number of nodes in each linked list is in the range [1, 100].
	0 <= Node.val <= 9
	It is guaranteed that the list represents a number that does not have leading zeros.


### 解析

根据题意，给定两个代表两个非负整数的非空链表。 这些数字以相反的顺序存储在链表中，并且它们的每个节点都包含一个数字。 将两个数字相加并将结果和作为链表返回，返回的结果总和结果链表也以数字相反的顺序存在链表里面即可。并且这两个数字不包含任何前导零，除了数字 0 本身之外。

其实这道题很简单，之所以难度为 Medium ，就是因为题目把对两个数字的相加过程结合到了链表之中，我们需要一边遍历两个链表上的值，一边对每一位上的两个值进行求和和进位运算，如果这两个基本的内容已经掌握，题目很容易就做出来了。


思路很简单：
初始化一个结果 result 节点和虚拟指针 d 节点，初始化一个进位存储器 carry 为 0 
当 l1 不为空或者 l2 不为空的时候进行 while 循环：

* 	遍历这两个链表对应位置上的数字，如果该位置为空则设置为 0 ，不为空则是用节点值进行计算，然后将两个值与进位 carry 相加对 10 取模运算可以得到该位置上的数字 p ，将其转化为 ListNode 类型的节点赋与虚拟指针 d.next ；同时将两个值与进位 carry 相加除以 10 运算可以得到新的 carry 
* 	将 d 指针往后移动一位
* 	如果 l1 不为空向后移动一位；l2 进行同样操作

while 结束之后，可能 carry 还大于 0 ，表示还有进位数字，所以新建一个节点存放 carry ，并将节点赋值给 d.next ，最后返回 result.next 即可。

时间复杂度只和最长的一个链表有关系，所以为 O(max(L1, L2)) ，空间复杂度和最长的链表长度有关系，并且可能有多一位的进位，所以为 O(max(L1, L2)+1) 。

### 解答
				

	class Solution(object):
	    def addTwoNumbers(self, l1, l2):
	        result = d = ListNode(-1)
	        carry = 0
	        while l1 or l2:
	            a = l1.val if l1 else 0
	            b = l2.val if l2 else 0
	            p = (a + b + carry) % 10
	            d.next = ListNode(p)
	            carry = (a + b + carry) // 10
	            d = d.next
	            l1 = l1.next if l1 else None
	            l2 = l2.next if l2 else None
	        if carry:
	            d.next = ListNode(carry)
	        return result.next
            	      
			
### 运行结果


	Runtime: 64 ms, faster than 77.45% of Python online submissions for Add Two Numbers.
	Memory Usage: 13.7 MB, less than 18.13% of Python online submissions for Add Two Numbers.


### 原题链接



https://leetcode.com/problems/add-two-numbers/

您的支持是我最大的动力
