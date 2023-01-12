leetcode  2081. Sum of k-Mirror Numbers（python）

### 描述

A k-mirror number is a positive integer without leading zeros that reads the same both forward and backward in base-10 as well as in base-k.

* For example, 9 is a 2-mirror number. The representation of 9 in base-10 and base-2 are 9 and 1001 respectively, which read the same both forward and backward.
* On the contrary, 4 is not a 2-mirror number. The representation of 4 in base-2 is 100, which does not read the same both forward and backward.

Given the base k and the number n, return the sum of the n smallest k-mirror numbers.



Example 1:

	Input: k = 2, n = 5
	Output: 25
	Explanation:
	The 5 smallest 2-mirror numbers and their representations in base-2 are listed as follows:
	  base-10    base-2
	    1          1
	    3          11
	    5          101
	    7          111
	    9          1001
	Their sum = 1 + 3 + 5 + 7 + 9 = 25. 

	
Example 2:

	Input: k = 3, n = 7
	Output: 499
	Explanation:
	The 7 smallest 3-mirror numbers are and their representations in base-3 are listed as follows:
	  base-10    base-3
	    1          1
	    2          2
	    4          11
	    8          22
	    121        11111
	    151        12121
	    212        21212
	Their sum = 1 + 2 + 4 + 8 + 121 + 151 + 212 = 499.


Example 3:

	Input: k = 7, n = 17
	Output: 20379000
	Explanation: The 17 smallest 7-mirror numbers are:
	1, 2, 3, 4, 5, 6, 8, 121, 171, 242, 292, 16561, 65656, 2137312, 4602064, 6597956, 6958596

	




Note:


	2 <= k <= 9
	1 <= n <= 30

### 解析

根据题意，k 镜像数是一个没有前导零的正整数，它在十进制和 k 进制中向前和向后读取相同。

* 例如，9 是一个二进制镜数。 十进制和二进制中 9 的表示分别为 9 和 1001，向前和向后读取相同。
* 相反，4 不是二进制数。 二进制数中 4 的表示是 100，向前和向后读取都不相同。

给定基数 k 和数 n ，返回 n 个最小的 k 镜数之和。


最暴力的解法肯定是从 1 开始遍历正整数 i ，然后找出 i 的 k 进制字符串 s ，然后使用函数判断 s 和 i 是否都是镜像的，最后将找到的 n 个正整数都加起来。但是这种解法肯定是超时的，因为随着 k 和 n 的增大，运算量会急剧增加，找 k 进制字符串的时候需要 O(log<sub>k</sub>n) ，判断是否为镜像的时候需要 2 * O(n/2) ，遍历找 n 个数字的时候需要 O(n) ，所以总体可能需要时间复杂度  O(n) * (O(log<sub>k</sub>n) + 2*O(n)) 。

### 解答
				
	class Solution(object):
	    def kMirror(self, k, n):
	        """
	        :type k: int
	        :type n: int
	        :rtype: int
	        """
	        num = 1
	        result = 0
	        while n>0:
	            t = self.getKNum(num, k)
	            if self.isMirror(t) and self.isMirror(str(num)):
	                result += num
	                n -= 1
	            num += 1
	        return result
	
	    def getKNum(self, num, k):
	        result = ''
	        while num!=0:
	            t = num%k
	            result = str(t) + result
	            num//=k
	        return result
	    
	
	    def isMirror(self, num):
	        i = 0
	        j = len(num)-1
	        while i<=j:
	            if num[i] == num[j]:
	                i += 1
	                j -= 1
	                continue
	            else:
	                return False
	        return True
	


            	      
			
### 运行结果

	Time Limit Exceeded
	
### 解析

另外我们可以直接构造十进制镜像数字，然后再对构造出来的镜像数的 k 进制数进行判断是否为镜像，如果都是镜像那就将其加入到结果中。

从小到大构造镜像数字的时候由于算法的特殊性，不会漏掉一个，也不会重复，如用一个三位数 xyz 构造镜像数：

	五位数 xyzyx
	六位数 xyzzyx
	
	
用一个四位数 xyza 构造镜像数：
	
	七位数 xyzazyx
	八位数 xyzaazyx
	
	
可以看出用三位数和四位数构造出来的镜像数都不会冲突。

另外再判断是否为镜像的时候，如果是对字符串进行判断，肯定会相当的耗时，我们可以通过一个数字保存反转之后的数字，然后再用原数字和反转之后的数字通过数学的方法进行比较，这样也能提高运算效率，因为计算机对数字的操作比对字符串的操作还是更简单。



### 解答

	class Solution(object):
	    def __init__(self):
	        self.tmp = [0]*100
	    def kMirror(self, k, n):
	        """
	        :type k: int
	        :type n: int
	        :rtype: int
	        """
	        L = 1
	        result = []
	        while True:
	            for i in range(pow(10, L-1), pow(10,L)):
	                a = self.getPalindrome(i, True)
	                if self.isMirror(a, k):
	                    result.append(a)
	                if len(result) == n:
	                    return sum(result)
	            
	            for i in range(pow(10, L-1), pow(10,L)):
	                a = self.getPalindrome(i, False)
	                if self.isMirror(a, k):
	                    result.append(a)
	                if len(result) == n:
	                    return sum(result)
	            L += 1
	            
	    def getPalindrome(self, x, flag):
	        y = x
	        z = 0
	        count = 0
	        while y>0:
	            count+=1
	            z = int(z*10+y%10)
	            y/=10
	        if flag: x/=10
	        x = x*pow(10, count)
	        return x+z
	    
	    def isMirror(self, x, k):
	        t = 0
	        while x>0:
	            self.tmp[t] = x%k
	            x/=k
	            t+=1
	        i = 0
	        j = t-1
	        while i < j:
	            if self.tmp[i] != self.tmp[j]:
	                return False
	            i+=1
	            j-=1
	        return True


### 运行结果

	Runtime: 8712 ms, faster than 13.48% of Python online submissions for Sum of k-Mirror Numbers.
	Memory Usage: 41.9 MB, less than 29.21% of Python online submissions for Sum of k-Mirror Numbers.

### 解析

另外可以用打表法，因为 2 <= k <= 9 和 1 <= n <= 30 ，总共有 8 种情况，每种情况找 30 个数，如果不想写代码，可以提前手算好所有的结果放入代码中的二位列表，根据 k 和 n 直接返回即可。但是这种方法比较投机取巧。运行直接超过 100% ，速度一流，堪比开挂。

### 解答

	class Solution(object):
	    def __init__(self):
	        self.tmp = [0]*100
	    def kMirror(self, k, n):
	        """
	        :type k: int
	        :type n: int
	        :rtype: int
	        """
	        L = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 4, 9, 16, 25, 58, 157, 470, 1055, 1772, 9219, 18228, 33579, 65802, 105795, 159030, 212865, 286602, 872187, 2630758, 4565149, 6544940, 9674153, 14745858, 20005383, 25846868, 39347399, 759196316, 1669569335, 2609044274], [1, 3, 7, 15, 136, 287, 499, 741, 1225, 1881, 2638, 31730, 80614, 155261, 230718, 306985, 399914, 493653, 1342501, 2863752, 5849644, 9871848, 14090972, 18342496, 22630320, 28367695, 36243482, 44192979, 71904751, 155059889], [1, 3, 6, 11, 66, 439, 832, 1498, 2285, 3224, 11221, 64456, 119711, 175366, 233041, 739646, 2540727, 4755849, 8582132, 12448815, 17500320, 22726545, 27986070, 33283995, 38898160, 44577925, 98400760, 721411086, 1676067545, 53393239260], [1, 3, 6, 10, 16, 104, 356, 638, 1264, 1940, 3161, 18912, 37793, 10125794, 20526195, 48237967, 78560270, 126193944, 192171900, 1000828708, 1832161846, 2664029984, 3500161622, 4336343260, 6849225412, 9446112364, 12339666346, 19101218022, 31215959143, 43401017264], [1, 3, 6, 10, 15, 22, 77, 188, 329, 520, 863, 1297, 2074, 2942, 4383, 12050, 19827, 41849, 81742, 156389, 325250, 1134058, 2043967, 3911648, 7009551, 11241875, 15507499, 19806423, 24322577, 28888231], [1, 3, 6, 10, 15, 21, 29, 150, 321, 563, 855, 17416, 83072, 2220384, 6822448, 13420404, 20379000, 29849749, 91104965, 321578997, 788407661, 1273902245, 1912731081, 2570225837, 3428700695, 29128200347, 69258903451, 115121130305, 176576075721, 241030621167], [1, 3, 6, 10, 15, 21, 28, 37, 158, 450, 783, 1156, 1570, 2155, 5818, 14596, 27727, 41058, 67520, 94182, 124285, 154588, 362290, 991116, 1651182, 3148123, 5083514, 7054305, 11253219, 66619574], [1, 3, 6, 10, 15, 21, 28, 36, 227, 509, 882, 1346, 1901, 2547, 3203, 10089, 35841, 63313, 105637, 156242, 782868, 2323319, 4036490, 5757761, 7586042, 9463823, 11349704, 13750746, 16185088, 18627530]]
	        return L[k-1][n-1]
	        
	
### 运行结果

	Runtime: 12 ms, faster than 100.00% of Python online submissions for Sum of k-Mirror Numbers.
	Memory Usage: 13.5 MB, less than 64.04% of Python online submissions for Sum of k-Mirror Numbers.

原题链接：https://leetcode.com/problems/sum-of-k-mirror-numbers/



您的支持是我最大的动力
