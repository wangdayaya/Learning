leetcode  1884. Egg Drop With 2 Eggs and N Floors（python）

### 描述


You are given two identical eggs and you have access to a building with n floors labeled from 1 to n.

You know that there exists a floor f where 0 <= f <= n such that any egg dropped at a floor higher than f will break, and any egg dropped at or below floor f will not break.

In each move, you may take an unbroken egg and drop it from any floor x (where 1 <= x <= n). If the egg breaks, you can no longer use it. However, if the egg does not break, you may reuse it in future moves.

Return the minimum number of moves that you need to determine with certainty what the value of f is.




Example 1:

	Input: n = 2
	Output: 2
	Explanation: We can drop the first egg from floor 1 and the second egg from floor 2.
	If the first egg breaks, we know that f = 0.
	If the second egg breaks but the first egg didn't, we know that f = 1.
	Otherwise, if both eggs survive, we know that f = 2.

	
Example 2:
	
	Input: n = 100
	Output: 14
	Explanation: One optimal strategy is:
	- Drop the 1st egg at floor 9. If it breaks, we know f is between 0 and 8. Drop the 2nd egg starting
	  from floor 1 and going up one at a time to find f within 7 more drops. Total drops is 1 + 7 = 8.
	- If the 1st egg does not break, drop the 1st egg again at floor 22. If it breaks, we know f is between 9
	  and 21. Drop the 2nd egg starting from floor 10 and going up one at a time to find f within 12 more
	  drops. Total drops is 2 + 12 = 14.
	- If the 1st egg does not break again, follow a similar process dropping the 1st egg from floors 34, 45,
	  55, 64, 72, 79, 85, 90, 94, 97, 99, and 100.
	Regardless of the outcome, it takes at most 14 drops to determine f.





Note:

	1 <= n <= 1000


### 解析


根据题意，题目给出了两个相同的鸡蛋，并且可以进入一栋有 n 层楼的建筑物，楼层的标记为从 1 到 n 。存在一个 f 层，其中 0 <= f <= n 使得任何高于 f 层的鸡蛋落下都会破裂，而任何从 f 层或以下的鸡蛋落下都不会破裂。

在每次移动中，我们可以拿起一个完整的鸡蛋并将其从任何楼层 x（其中 1 <= x <= n）掉落。 如果鸡蛋破了，就不能再使用了。 但是如果鸡蛋没有破裂，我们可以在以后重新使用它。返回确定 f 的值所需的最小移动次数。

我们可以找规律：

* 如果 100 层楼，我们现在有一个鸡蛋，只能从一楼往上一层一层慢慢扔，那我们最坏情况要扔 100 次；
* 如果我们有两个鸡蛋，使用类似二分法的方式扔，第一种情况如果在 50 楼扔摔破了一个，那么只能用第二个蛋从一楼往上一层一层开始从下往上扔，第二种情况在 50 楼没有摔破，继续用二分法找楼层扔鸡蛋，总之那么最坏情况扔 1+49 次；
* 最坏情况下移动最少的次数就是答案，也就是我们扔的鸡蛋次数最少，如果我们选择从 10 楼开始扔，如果摔破了，用第二个蛋从 1 楼往上一层一层找，如果没有摔破，用第一个蛋开始从 20 楼开始扔，继续这个过程，最坏的情况是第一个鸡蛋扔到 100 层仍然完好，然后从第 91 层开始一层一层上楼扔，最差需要 10+9 次；
* 我们再优化策略，假如我们选择从最佳的楼层 n 楼开始扔鸡蛋，只要第一个鸡蛋不破，就往上走 n-1 层，也就是在 2n-1 层重新扔鸡蛋，这样如果第一个鸡蛋摔破，我们用第二个鸡蛋检查的前面的 n-1 层，如果没有碎继续往上走 n-2 层，也就是在 3n-3 楼扔鸡蛋，一直循环这个过程，那就是 n + (n-1) + (n-2) + (n-3) + (n-4) + … + 1 >= 100 ，即 n (n+1) / 2 >= 100 ，结果向上取整 n 为 14 ，也就是最差情况需要扔 14 次。也就是第一次在 14 楼扔的蛋就碎了，然后用第二个蛋从 1 楼开始往上找。
* 所以只需要通过公式计算即即可找出答案。

其实上面整个推理过程也是我看的网上大神的，自己理解的不是很深刻，只知道写的代码误打误撞是通过了。

### 解答
				
	class Solution(object):
	    def twoEggDrop(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        for x in range(n+1):
	            if  x * (x+1) // 2 >= n :
	                return x
	        
	        

            	      
			
### 运行结果

	Runtime: 8 ms, faster than 99.26% of Python online submissions for Egg Drop With 2 Eggs and N Floors.
	Memory Usage: 13.4 MB, less than 77.78% of Python online submissions for Egg Drop With 2 Eggs and N Floors.


原题链接：https://leetcode.com/problems/egg-drop-with-2-eggs-and-n-floors/

特别感谢：https://blog.csdn.net/a130737/article/details/44751691

您的支持是我最大的动力
