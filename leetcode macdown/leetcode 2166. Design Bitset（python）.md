leetcode 2166. Design Bitset （python）

### 前言

这是 Weekly Contest 279 比赛的第三题，难度 Medium ，考察的是对位运算，这道题说难也难，说不难也不难，看你有没有找到诀窍，如果用位运算是有点难度，如果用数组的方法倒也简单。


### 描述


A Bitset is a data structure that compactly stores bits.

Implement the Bitset class:

* Bitset(int size) Initializes the Bitset with size bits, all of which are 0.
* void fix(int idx) Updates the value of the bit at the index idx to 1. If the value was already 1, no change occurs.
* void unfix(int idx) Updates the value of the bit at the index idx to 0. If the value was already 0, no change occurs.
* void flip() Flips the values of each bit in the Bitset. In other words, all bits with value 0 will now have value 1 and vice versa.
* boolean all() Checks if the value of each bit in the Bitset is 1. Returns true if it satisfies the condition, false otherwise.
* boolean one() Checks if there is at least one bit in the Bitset with value 1. Returns true if it satisfies the condition, false otherwise.
* int count() Returns the total number of bits in the Bitset which have value 1.
* String toString() Returns the current composition of the Bitset. Note that in the resultant string, the character at the i<sup>th</sup> index should coincide with the value at the i<sup>th</sup> bit of the Bitset.


Example 1:

	Input
	["Bitset", "fix", "fix", "flip", "all", "unfix", "flip", "one", "unfix", "count", "toString"]
	[[5], [3], [1], [], [], [0], [], [], [0], [], []]
	Output
	[null, null, null, null, false, null, null, true, null, 2, "01010"]
	
	Explanation
	Bitset bs = new Bitset(5); // bitset = "00000".
	bs.fix(3);     // the value at idx = 3 is updated to 1, so bitset = "00010".
	bs.fix(1);     // the value at idx = 1 is updated to 1, so bitset = "01010". 
	bs.flip();     // the value of each bit is flipped, so bitset = "10101". 
	bs.all();      // return False, as not all values of the bitset are 1.
	bs.unfix(0);   // the value at idx = 0 is updated to 0, so bitset = "00101".
	bs.flip();     // the value of each bit is flipped, so bitset = "11010". 
	bs.one();      // return True, as there is at least 1 index with value 1.
	bs.unfix(0);   // the value at idx = 0 is updated to 0, so bitset = "01010".
	bs.count();    // return 2, as there are 2 bits with value 1.
	bs.toString(); // return "01010", which is the composition of bitset.





Note:


	1 <= size <= 10^5
	0 <= idx <= size - 1
	At most 10^5 calls will be made in total to fix, unfix, flip, all, one, count, and toString.
	At least one call will be made to all, one, count, or toString.
	At most 5 calls will be made to toString.

### 解析


根据题意，实现一种名为 Bitset 的的数据结构。实现 Bitset 类有以下函数：

* Bitset(int size) 用 size 个全为 0 的位来初始化 Bitset 
* void fix(int idx) 将索引 idx 处的位值更新为 1
* void unfix(int idx) 将索引 idx 处的位值更新为 0
* void flip() 翻转 Bitset 中每个位的值
* boolean all() 检查 Bitset 中每个位的值是否都为 1
* boolean one() 检查 Bitset 中是否至少有一位值为 1
* int count() 返回 Bitset 中值为 1 的总位数
* String toString() 返回 Bitset 的当前字符串组合

这道题的难点在于限制条件中的一句话：

	At most 10^5 calls will be made in total to fix, unfix, flip, all, one, count, and toString.
	
也就是所有的操作都可能经历最多 10^5 的调用，其实这也就是从侧面告诉我们，这些函数的实现的时间复杂度都要在 O(1) 的范围内，其实从题目一开始定义这个类叫 Bitset 的时候，也其实相当于告诉了我们肯定要用位运算，而且位运算效率很高，也足可以满足时间复杂度的要求。

首先我们定义一个计数器 countOne ，专门统计出现的 1 的个数，这样我们就能在调用 all 、one 、count 这三个函数的时候直接进行判断和返回。而 countOne 的变化取决于 fix 、unfix 、 flip 三个函数的调用变化。

其次我们定义一个整数 result 为 0 ，然后在调用 fix 函数的时候，判断 result & (1 << idx) ，如果为 0 ，说明该位置上是 0 ，我们要调用按位或运算将该位置上的数字设置为 1 ，同时 countOne 加一。

在调用 unfix 函数的时候，我们判断 self.result & (1 << idx) ，如果为 1 ，说明该位置上是 1 ，我们要调用按位与运算将该位置上的数字设置为 0 ，同时  countOne 减一。

在调用  flip 的时候，我们要使用按位与运算，将所有数字都反转，1 变成 0 ，0 变成 1 ，同时更新 countOne 。

在调用 toString 的时候，因为我们从一开始整个的操作都是从右向左的顺序，现在我们要将 result 对应的二进制反转成从左向右的顺序，同时用 0 在右边末尾补齐，试长度保证为 size 。

fix 、 unfix 、flip 都是位运算所以时间复杂度为  O(1) ，all 、 one 、count 直接进行判断返回，所以时间复杂度也是 O(1) 。toString 函数的时间复杂度为 O(N) ，而且限制条件中也说明了最多调用 5 次  toString ，所以时间复杂度可以满足。


### 解答
				

	class Bitset(object):
	
	    def __init__(self, size):
	        self.result = 0
	        self.length = size
	        self.countOne = 0
	
	    def fix(self, idx):
	        if self.result & (1 << idx) == 0:
	            self.result |= 1 << idx
	            self.countOne += 1
	
	    def unfix(self, idx):
	        if self.result & (1 << idx):
	            self.result ^= 1 << idx
	            self.countOne -= 1
	
	    def flip(self):
	        self.result ^= (1 << self.length) - 1
	        self.countOne = self.length - self.countOne
	
	    def all(self):
	        return self.countOne == self.length
	
	    def one(self):
	        return self.result > 0
	
	    def count(self):
	        return self.countOne
	
	    def toString(self):
	        result = bin(self.result)[2:]
	        return result[::-1] + '0' * (self.length - len(result))
            	      

	
### 运行结果


	48 / 48 test cases passed.
	Status: Accepted
	Runtime: 1605 ms
	Memory Usage: 47.4 MB


### 解析

其实这道题最关键的就是解决 fix 、unfix 、flip 这三个函数，而对于前两个函数其实就是在索引位置上进行赋值，所以用列表是最方便的，省时省力，但是如果使用列表，在调用 flip 函数的时候如果按位置去反转就会超时，这里我们可以取巧，因为没有空间复杂度的限制，我们定义一个结果列表 result ，另外定义一个与 result 相反转的列表 fliped ，这样我们在执行 fix 和 unfix 的时候对于 result 和 fliped 两个列表进行相反的操作，方便我们在调用 flip 函数的时候只需要互换值就能完成函数功能，是不是很妙！至于其他的函数操作和上面的思路一样，最后在调用 toString 的时候，将 result 从左到右连接成字符串返回即可。

这样所有函数的时间复杂度都是 O(1) ，整个 Bitset 类的空间复杂度为 O(2\*N) ，满足题目要求。从耗时来看比上面的方法要节省一半的时间。

### 解答
	
	class Bitset(object):
	
	    def __init__(self, size):
	        self.result = [0] * size
	        self.fliped = [1] * size
	        self.countOne = 0
	        self.length = size
	
	    def fix(self, idx):
	        if self.result[idx] == 0:
	            self.countOne += 1
	            self.result[idx] = 1
	            self.fliped[idx] = 0
	
	    def unfix(self, idx):
	        if self.result[idx] == 1:
	            self.countOne -= 1
	            self.result[idx] = 0
	            self.fliped[idx] = 1
	
	    def flip(self):
	        self.result, self.fliped = self.fliped, self.result
	        self.countOne = self.length - self.countOne
	
	    def all(self):
	        return self.countOne == self.length
	
	    def one(self):
	        return self.countOne > 0
	
	    def count(self):
	        return self.countOne
	
	    def toString(self):
	        return ''.join(map(str,self.result))

### 运行结果


	48 / 48 test cases passed.
	Status: Accepted
	Runtime: 782 ms
	Memory Usage: 48.9 MB

### 原题链接


https://leetcode.com/contest/weekly-contest-279/problems/design-bitset/

您的支持是我最大的动力
