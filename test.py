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
    
dict = {"1": 1,
        "2": 2}

print(len(dict))

