本来是想自己写但是复杂度太强，偶然搜到 python 中有一个库 bisect 可以实现真的是太妙了 。

基本用法，使用起来很方便，其实就是在找可以插入的位置，但是不会真的插入：

    import bisect 

    l = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100] 
    bisect.bisect(l, 55) # returns 7 
    
耗时测试，从结果来看 bisect 的速度最快：

    import timeit 
    import bisect

    print(timeit.timeit('bisect.bisect([1, 4, 9, 16, 25, 36, 49, 64, 81, 100], 55)','import bisect'))  
    # 0.2991909169941209
    print(timeit.timeit('next(i for i,n in enumerate([1, 4, 9, 16, 25, 36, 49, 64, 81, 100]) if n > 55)'))  
    # 0.7511563330190256
    print(timeit.timeit('next(([1, 4, 9, 16, 25, 36, 49, 64, 81, 100].index(n) for n in [1, 4, 9, 16, 25, 36, 49, 64, 81, 100] if n > 55))')) 
    # 0.6965440830099396

另外还有两种常用方法，bisect_left 和 bisect_right 函数，用入处理将会插入重复数值的情况，返回将会插入的位置。

- bisect_left(seq, x) x存在时返回x左侧的位置
- bisect_right(seq, x) x存在时返回x右侧的位置

举例：

    print(bisect.bisect_left([2,4,7,9],2)) 
    # 0
    print(bisect.bisect_left([2,4,7,9],3)) 
    # 1
    print(bisect.bisect_right([2,4,7,9],2)) 
    # 1
    print(bisect.bisect_right([2,4,7,9],3)) 
    # 1