leetcode  2241. Design an ATM Machine（python）

这道题是第 76 场 leetcode 双周赛的第三题，难度为 Medium ，主要考察的是数学中的解方程问题


### 描述


There is an ATM machine that stores banknotes of 5 denominations: 20, 50, 100, 200, and 500 dollars. Initially the ATM is empty. The user can use the machine to deposit or withdraw any amount of money.

When withdrawing, the machine prioritizes using banknotes of larger values.

* For example, if you want to withdraw $300 and there are 2 $50 banknotes, 1 $100 banknote, and 1 $200 banknote, then the machine will use the $100 and $200 banknotes.
* However, if you try to withdraw $600 and there are 3 $200 banknotes and 1 $500 banknote, then the withdraw request will be rejected because the machine will first try to use the $500 banknote and then be unable to use banknotes to complete the remaining $100. Note that the machine is not allowed to use the $200 banknotes instead of the $500 banknote.

Implement the ATM class:

* ATM() Initializes the ATM object.
* void deposit(int[] banknotesCount) Deposits new banknotes in the order $20, $50, $100, $200, and $500.
* int[] withdraw(int amount) Returns an array of length 5 of the number of banknotes that will be handed to the user in the order $20, $50, $100, $200, and $500, and update the number of banknotes in the ATM after withdrawing. Returns [-1] if it is not possible (do not withdraw any banknotes in this case).



Example 1:

	Input
	["ATM", "deposit", "withdraw", "deposit", "withdraw", "withdraw"]
	[[], [[0,0,1,2,1]], [600], [[0,1,0,1,1]], [600], [550]]
	Output
	[null, null, [0,0,1,0,1], null, [-1], [0,1,0,0,1]]
	
	Explanation
	ATM atm = new ATM();
	atm.deposit([0,0,1,2,1]); // Deposits 1 $100 banknote, 2 $200 banknotes,
	                          // and 1 $500 banknote.
	atm.withdraw(600);        // Returns [0,0,1,0,1]. The machine uses 1 $100 banknote
	                          // and 1 $500 banknote. The banknotes left over in the
	                          // machine are [0,0,0,2,0].
	atm.deposit([0,1,0,1,1]); // Deposits 1 $50, $200, and $500 banknote.
	                          // The banknotes in the machine are now [0,1,0,3,1].
	atm.withdraw(600);        // Returns [-1]. The machine will try to use a $500 banknote
	                          // and then be unable to complete the remaining $100,
	                          // so the withdraw request will be rejected.
	                          // Since the request is rejected, the number of banknotes
	                          // in the machine is not modified.
	atm.withdraw(550);        // Returns [0,1,0,0,1]. The machine uses 1 $50 banknote
	                          // and 1 $500 banknote.





Note:

	banknotesCount.length == 5
	0 <= banknotesCount[i] <= 10^9
	1 <= amount <= 10^9
	At most 5000 calls in total will be made to withdraw and deposit.
	At least one call will be made to each function withdraw and deposit.


### 解析


根据题意，有一台 ATM 机可以存储 5 种面额的纸币：20、50、100、200 和 500 美元。 最初，ATM 是空的。 用户可以使用机器存入或提取任意数量的钱。取款时，机器优先使用面值较大的纸币，例如：

如果提取 300 美元，现在有 2 张 50 美元、1 张 100 美元和 1 张 200 美元的钞票，那么机器将使用 100 美元和 200 美元的钞票；但是，如果提取 600 美元并且现在有 3 张 200 美元钞票和 1 张 500 美元钞票，则取款请求将被拒绝，因为机器将首先尝试使用 500 美元钞票，然后无法使用钞票完成剩余的 100 美元。

这道题其实就是考察贪心，我们在取钞票的时候，尽量先取面额大的钞票，这样才能满足题意，初始化一个表示面额的列表 m 和一个对应表示相应面额数量的列表 L 。

在进行存钱的时候，调用 deposit ，我们只需要遍历这个参数 banknotesCount ，将对应的数字加入到 L 对应的位置即可。

在进行取钱的时候，需要定义一个操作列表 result 表示为了取 amount 元需要取出的对应面额的数量，然后从大面额遍历到小面额，计算取每种面额的数量，当 amount 为 0 的时候，将原始的 L 和 result 对应位置上面的数量相减，表示 ATM 中各个面额剩余的数量，最后将 result 返回即可，遍历结束仍不能正常返回，说明无法进行取钱的操作，直接返回 [-1] 即可。

deposit 和 withdraw 的时间复杂度几乎都为 O(5) ，空间复杂度为 O(5) ，所以可以满足限制条件中至少 5000 次的调用要求。

### 解答
				

	class ATM(object):
	
	    def __init__(self):
	        self.L = [0, 0, 0, 0, 0]
	        self.m = [20, 50, 100, 200 ,500]
	
	    def deposit(self, banknotesCount):
	        """
	        :type banknotesCount: List[int]
	        :rtype: None
	        """
	        for i, b in enumerate(banknotesCount):
	            self.L[i] += b
	    
	    def withdraw(self, amount):
	        """
	        :type amount: int
	        :rtype: List[int]
	        """
	        result = [0,0,0,0,0]
	        for i in range(4, -1, -1):
	            n = min(self.L[i], amount // self.m[i])
	            result[i] += n
	            amount -= n * self.m[i]
	            if amount == 0:
	                self.L = [a-b for a,b in zip(self.L, result)]
	                return result
	        return [-1]
  
			
### 运行结果

	
	152 / 152 test cases passed.
	Status: Accepted
	Runtime: 562 ms
	Memory Usage: 17 MB


### 原题链接


https://leetcode.com/contest/biweekly-contest-76/problems/design-an-atm-machine/


您的支持是我最大的动力
