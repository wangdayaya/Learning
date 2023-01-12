leetcode  2347. Best Poker Hand（python）




### 描述

You are given an integer array ranks and a character array suits. You have 5 cards where the ith card has a rank of ranks[i] and a suit of suits[i]. The following are the types of poker hands you can make from best to worst:

* "Flush": Five cards of the same suit.
* "Three of a Kind": Three cards of the same rank.
* "Pair": Two cards of the same rank.
* "High Card": Any single card.

Return a string representing the best type of poker hand you can make with the given cards. Note that the return values are case-sensitive.



Example 1:

	Input: ranks = [13,2,3,1,9], suits = ["a","a","a","a","a"]
	Output: "Flush"
	Explanation: The hand with all the cards consists of 5 cards with the same suit, so we have a "Flush".

	
Example 2:

	Input: ranks = [4,4,2,4,4], suits = ["d","a","a","b","c"]
	Output: "Three of a Kind"
	Explanation: The hand with the first, second, and fourth card consists of 3 cards with the same rank, so we have a "Three of a Kind".
	Note that we could also make a "Pair" hand but "Three of a Kind" is a better hand.
	Also note that other cards could be used to make the "Three of a Kind" hand.


Example 3:

	Input: ranks = [10,10,2,12,9], suits = ["a","b","c","a","d"]
	Output: "Pair"
	Explanation: The hand with the first and second card consists of 2 cards with the same rank, so we have a "Pair".
	Note that we cannot make a "Flush" or a "Three of a Kind".



Note:

	ranks.length == suits.length == 5
	1 <= ranks[i] <= 13
	'a' <= suits[i] <= 'd'
	No two cards have the same rank and suit.


### 解析

根据题意，给定一个整数数组 ranks 和一个字符数组 suits 。 有 5 张牌，其中第 i 张牌的等级为 ranks[i] 和一套花色 suits[i] 。 以下是从最好到最差的扑克牌类型：

* “同花顺”：五张同花色的牌。
* “三类”：三张相同等级的牌。
* “对子”：两张相同等级的牌。
* “高牌”：任何单张牌。

返回一个字符串，表示可以使用给定卡片最佳扑克类型。 请注意，返回值区分大小写。

这道题是比较简单的，只要按照题目的意思写代码即可。时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def bestHand(self, ranks, suits):
	        """
	        :type ranks: List[int]
	        :type suits: List[str]
	        :rtype: str
	        """
	        N = len(ranks)
	        if len(set(suits)) == 1:
	            return "Flush"
	        c = collections.Counter(ranks)
	        if 4 in c.values() or 3 in c.values():
	            return "Three of a Kind"
	        if 2 in c.values():
	            return "Pair"
	        if len(set(ranks)) == N:
	            return "High Card"

### 运行结果
	
	93 / 93 test cases passed.
	Status: Accepted
	Runtime: 16 ms
	Memory Usage: 13.5 MB


### 原题链接

https://leetcode.com/contest/biweekly-contest-83/problems/best-poker-hand/


您的支持是我最大的动力
