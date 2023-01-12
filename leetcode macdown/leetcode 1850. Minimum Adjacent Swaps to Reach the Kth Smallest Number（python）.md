

### 描述


You are given a string num, representing a large integer, and an integer k.

We call some integer wonderful if it is a permutation of the digits in num and is greater in value than num. There can be many wonderful integers. However, we only care about the smallest-valued ones.

For example, when num = "5489355142":

* The 1st smallest wonderful integer is "5489355214".
* The 2nd smallest wonderful integer is "5489355241".
* The 3rd smallest wonderful integer is "5489355412".
* The 4th smallest wonderful integer is "5489355421".


Return the minimum number of adjacent digit swaps that needs to be applied to num to reach the kth smallest wonderful integer.

The tests are generated in such a way that kth smallest wonderful integer exists.

 


Example 1:


	Input: num = "5489355142", k = 4
	Output: 2
	Explanation: The 4th smallest wonderful number is "5489355421". To get this number:
	- Swap index 7 with index 8: "5489355142" -> "5489355412"
	- Swap index 8 with index 9: "5489355412" -> "5489355421"
	
Example 2:


	Input: num = "11112", k = 4
	Output: 4
	Explanation: The 4th smallest wonderful number is "21111". To get this number:
	- Swap index 3 with index 4: "11112" -> "11121"
	- Swap index 2 with index 3: "11121" -> "11211"
	- Swap index 1 with index 2: "11211" -> "12111"
	- Swap index 0 with index 1: "12111" -> "21111"

Example 3:

		
	Input: num = "00123", k = 1
	Output: 1
	Explanation: The 1st smallest wonderful number is "00132". To get this number:
	- Swap index 3 with index 4: "00123" -> "00132"
	


Note:

	
	2 <= num.length <= 1000
	1 <= k <= 1000
	num only consists of digits.

### 解析

根据题意，给出了一个包含了数字字符的字符串 num ，要求我们计算经过多少次将相邻的两个字符进行交换的操作，可以得到一个比 num 的数字大的第 k 个字符串。

其实可以看出来这个题就是 [31. Next Permutation](https://leetcode.com/problems/next-permutation/) 的升级版本，找比 num 大的第 k 的字符串组合，直接调用之前的函数 nextPermutation ，经过稍微的调整即可拿来使用，关键就在于得到第 k 大的字符串的结果之后和最开始的 num 比较进行了多少次相邻字符的交换。

考虑到本题中字符串个数不大于 1000 ，暴力的冒泡贪心 O(N^2) 也能接受。具体方法是：对于 new[i] ，我们从 old[0] 开始找起、直到找到第一个 old[j]==new[i] ，那么从 0 到 j 的过程中所遇到的尚未匹配的字符个数（非 # 的字符个数），就是需要的 adjacent swap 次数。注意对于匹配好的 new[j] ，我们立即将其替换为'#'作为标记。比如说：

* old : 55142
* new: 55214
* step 1: old -> #5142 +0 
* step 2: old -> ##142 +2
* step 3: old -> ##14# +0
* step 4: old -> ###4# +0
* step 5: old -> ##### +0
* 所以总共 2 次 swap 将 old 变成 new : 0+2+0+0+0=2

从最后的结果看估计刚好在超时的边缘徘徊，幸好通过了。
### 解答
				
	
	class Solution(object):
	    def getMinSwaps(self, num, k):
	        """
	        :type num: str
	        :type k: int
	        :rtype: int
	        """
	        tmp = list(num)
	        for i in range(k):
	            num = self.nextPermutation(num)
	        result = 0
	        for i in range(len(num)):
	            count = 0
	            for j in range(len(tmp)):
	                if tmp[j] == num[i]:
	                    tmp[j] = '#'
	                    break
	                if tmp[j] == '#':
	                    continue
	                count += 1
	            result += count
	        return result
	        
	        
	        
	    def nextPermutation(self, num):
	        num = list(num)
	        i = len(num)-1
	        while i>0:
	            if num[i-1]<num[i]:
	                break
	            i = i-1
	        i = i-1
	        j = len(num)-1
	        while j>i:
	            if num[j]>num[i]:
	                break
	            j=j-1
	        num[i],num[j]=num[j],num[i]  
	        num[i+1:]=sorted(num[i+1:])
	        return ''.join(num)
	            
            	      
			
### 运行结果

	Runtime: 3812 ms, faster than 8.70% of Python online submissions for Minimum Adjacent Swaps to Reach the Kth Smallest Number.
	Memory Usage: 13.7 MB, less than 39.13% of Python online submissions for Minimum Adjacent Swaps to Reach the Kth Smallest Number.


### 解析


其实耗时主要在 getMinSwaps 函数这里，进行优化即可。

### 解答

	class Solution(object):
	    def getMinSwaps(self, num, k):
	        """
	        :type num: str
	        :type k: int
	        :rtype: int
	        """
	        tmp = list(num)
	        num = list(num)
	        for i in range(k):
	            num = self.nextPermutation(num)
	        return self.count(tmp, num)
	
	
	    def nextPermutation(self, num):
	        i = len(num)-1
	        while i>0:
	            if num[i-1]<num[i]:
	                break
	            i = i-1
	        i = i-1
	        j = len(num)-1
	        while j>i:
	            if num[j]>num[i]:
	                break
	            j=j-1
	        num[i],num[j]=num[j],num[i]  
	        num[i+1:]=sorted(num[i+1:])
	        return num
	    
	    def count(self, origin, num):
	        n = len(origin)
	        result = 0
	        for i in range(n):
	            if origin[i] != num[i]:
	                tmp = i+1
	                while origin[i]!=num[tmp]:
	                    tmp+=1
	                while i!=tmp:
	                    num[tmp], num[tmp-1] = num[tmp-1],num[tmp]
	                    tmp-=1
	                    result+=1
	        return result

### 运行结果

    Runtime: 1552 ms, faster than 69.57% of Python online submissions forMinimum Adjacent Swaps to Reach the Kth Smallest Number.
    Memory Usage: 13.4 MB, less than 100.00% of Python online submissions forMinimum Adjacent Swaps to Reach the Kth Smallest Number.



原题链接：https://leetcode.com/problems/minimum-adjacent-swaps-to-reach-the-kth-smallest-number/



您的支持是我最大的动力
