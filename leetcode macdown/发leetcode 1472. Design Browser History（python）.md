leetcode  1472. Design Browser History（python）

### 描述

You have a browser of one tab where you start on the homepage and you can visit another url, get back in the history number of steps or move forward in the history number of steps.

Implement the BrowserHistory class:

* BrowserHistory(string homepage) Initializes the object with the homepage of the browser.
* void visit(string url) Visits url from the current page. It clears up all the forward history.
* string back(int steps) Move steps back in history. If you can only return x steps in the history and steps > x, you will return only x steps. Return the current url after moving back in history at most steps.
* string forward(int steps) Move steps forward in history. If you can only forward x steps in the history and steps > x, you will forward only x steps. Return the current url after forwarding in history at most steps.




Example 1:


	Input:
	["BrowserHistory","visit","visit","visit","back","back","forward","visit","forward","back","back"]
	[["leetcode.com"],["google.com"],["facebook.com"],["youtube.com"],[1],[1],[1],["linkedin.com"],[2],[2],[7]]
	Output:
	[null,null,null,null,"facebook.com","google.com","facebook.com",null,"linkedin.com","google.com","leetcode.com"]
	
	Explanation:
	BrowserHistory browserHistory = new BrowserHistory("leetcode.com");
	browserHistory.visit("google.com");       // You are in "leetcode.com". Visit "google.com"
	browserHistory.visit("facebook.com");     // You are in "google.com". Visit "facebook.com"
	browserHistory.visit("youtube.com");      // You are in "facebook.com". Visit "youtube.com"
	browserHistory.back(1);                   // You are in "youtube.com", move back to "facebook.com" return "facebook.com"
	browserHistory.back(1);                   // You are in "facebook.com", move back to "google.com" return "google.com"
	browserHistory.forward(1);                // You are in "google.com", move forward to "facebook.com" return "facebook.com"
	browserHistory.visit("linkedin.com");     // You are in "facebook.com". Visit "linkedin.com"
	browserHistory.forward(2);                // You are in "linkedin.com", you cannot move forward any steps.
	browserHistory.back(2);                   // You are in "linkedin.com", move back two steps to "facebook.com" then to "google.com". return "google.com"
	browserHistory.back(7);                   // You are in "google.com", you can move back only one step to "leetcode.com". return "leetcode.com"


Note:

	1 <= homepage.length <= 20
	1 <= url.length <= 20
	1 <= steps <= 100
	homepage and url consist of  '.' or lower case English letters.
	At most 5000 calls will be made to visit, back, and forward.


### 解析

根据题意，就是让我们实现一下页面的浏览，前进，后退等操作，具体的定义题目中都已经给出来了，其实就是考察的栈的压栈和弹出的基本操作，只不过结合了实际的业务场景，用栈来操作具体的页面顺序，


* BrowserHistory 初始化相当于我们用显示器打开指定网址
* visit 函数相当于我们从当前页面跳转到新的指定的网址，且其前面的页面都清空，当前页面不能前进，只能倒退
* back 函数相当于页面后退 step 步，同时返回到达页面的网址字符串，如果 step 太大则只退到最开始的页面即可
* forward 函数相当于页面前进 step 步，同时返回到达页面的网址字符串，如果 step 太大则只前进到最后面的页面即可，


### 解答
					
	class BrowserHistory(object):
	    def __init__(self, homepage):
	        """
	        :type homepage: str
	        """
	        self.stack = [homepage]
	        self.index = 0
	
	    def visit(self, url):
	        """
	        :type url: str
	        :rtype: None
	        """
	        for _ in range(len(self.stack)-self.index-1):
	            self.stack.pop()
	        self.stack.append(url)
	        self.index += 1
	            
	
	    def back(self, steps):
	        """
	        :type steps: int
	        :rtype: str
	        """
	        if self.stack:
	            self.index = max(0, self.index-steps)
	            return self.stack[self.index]
	
	    def forward(self, steps):
	        """
	        :type steps: int
	        :rtype: str
	        """
	        if self.stack:
	            self.index = min(len(self.stack)-1, self.index+steps)
	            return self.stack[self.index]

            	      
			
### 运行结果


	Runtime: 282 ms, faster than 43.41% of Python online submissions for Design Browser History.
	Memory Usage: 16.5 MB, less than 37.21% of Python online submissions for Design Browser History.

原题链接：https://leetcode.com/problems/design-browser-history/



您的支持是我最大的动力
