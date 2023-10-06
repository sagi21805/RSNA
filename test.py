class MyList(list):
       
    def __add__(self, obj):
        r = MyList([])
        if type(obj) == list or type(obj) == MyList:
            for i, j in zip(self, obj):
                r.append(i + j)
        
        if type(obj) == int:
            for i in self:
                r.append(i + obj)
                
        return r
    
def f(i):
    if i <= 1:
        return i
    
    return i + f(i-1)
   

print(f(4))

