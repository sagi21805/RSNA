class MyList(list):
       
    def __add__(self, obj):
        r = MyList([])
        if type(obj) == list or type(obj) == MyList:
            for i, j in zip(self, obj):
                r.append(i + j)
        
        if type(obj) == int:
            r = MyList([])
            for i in self:
                r.append(i + obj)
                
        return r
            
l = MyList([1, 2, 3])
k = MyList([2, 3, 4])
j = [1, 2, 3]

print(k+ 1)