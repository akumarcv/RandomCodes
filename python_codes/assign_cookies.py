def find_content_children(greed_factors, cookie_sizes):
    
    greed_factors.sort()
    cookie_sizes.sort()
    
    greed, cookie= 0, 0
    count = 0
    while greed<len(greed_factors) and cookie<len(cookie_sizes):
        if greed_factors[greed]<=cookie_sizes[cookie]:
            count+=1
            greed+=1
            cookie+=1
        else:
            cookie+=1
            
    return count
def main():
  
    greed_factors = [
        [1, 2, 3],
        [10, 20, 30, 40 ,50 ,60 ,70, 80],     
        [3, 4, 5, 6, 7, 8],  
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],  
        [10, 9, 8, 7],
        [1000, 996, 867, 345, 23, 12]
    ]
    
    cookie_sizes = [
        [1, 1],     
        [10, 20, 30, 40 ,50 ,60 ,70, 80, 90, 100],  
        [1, 2],     
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],  
        [5, 6, 7, 8],
        []
    ]
    
    for i in range(len(greed_factors)):
        result = find_content_children(greed_factors[i], cookie_sizes[i])
        print(i + 1,".\tGreed factors:", greed_factors[i])
        print("\tCookie sizes:", cookie_sizes[i])
        print("\n\tResult:", result)
        print("-"*100)

main()