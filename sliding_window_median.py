from heapq import heappop, heappush, heapify

def median_sliding_window(nums, k):

    result = []
    min_heap = []
    max_heap = []
    outgoing_num = {}
    
    for i in range(k):
        heappush(max_heap, -nums[i])
    for i in range(0, k//2):
        element = -heappop(max_heap)
        heappush(min_heap, element)
    
    balance = 0
    i = k
    
    while True:
        # calculate median as the top element of the two heaps 
        if k % 2 == 0:
            result.append((-max_heap[0] + min_heap[0]) / 2)
        else:
            result.append(-max_heap[0])
            
        if i == len(nums):
            break
        
        # find the outgoing number and incoming number
        number_going_out = nums[i - k]
        number_coming_in = nums[i]
        
        # increment i by 1
        i+=1
        
        # check if the outgoing number is in min heap or max heap and update the balance
        if number_going_out <= -max_heap[0]:
            balance = balance -1
        else: 
            balance = balance + 1
        
        # add the outgoing numebr to the outgoing_num dictionary
        if number_going_out in outgoing_num:
            outgoing_num[number_going_out] += 1
        else:
            outgoing_num[number_going_out] = 1
            
        # add incoming number and update the balance accordingly
        if not max_heap or number_coming_in<=-max_heap[0]:
            heappush(max_heap, -number_coming_in)
            balance = balance + 1
        else:
            heappush(min_heap, number_coming_in)
            balance = balance - 1
        
        # rebalance the heaps
        if balance>0: 
            heappush(min_heap, -heappop(max_heap))
            balance -= 1
        if balance<0:
            heappush(max_heap, -heappop(min_heap))
            balance += 1
        balance = 0
        
        #remove the outgoign number from the heap if it is at the top 
        while -max_heap[0] in outgoing_num and outgoing_num[-max_heap[0]]>0:
            outgoing_num[-heappop(max_heap)] -= 1
        while min_heap and min_heap[0] in outgoing_num and outgoing_num[min_heap[0]]>0:
            outgoing_num[heappop(min_heap)] -= 1
    return result
  
        
def main():
    input = (
            ([3, 1, 2, -1, 0, 5, 8],4), 
            ([1, 2], 1), 
            ([4, 7, 2, 21], 2), 
            ([22, 23, 24, 56, 76, 43, 121, 1, 2, 0, 0, 2, 3, 5], 5), 
            ([1, 1, 1, 1, 1], 2))
    x = 1
    for i in input:
        print(x, ".\tInput array: ", i[0],  ", k = ", i[1], sep = "")
        print("\tMedians: ", median_sliding_window(i[0], i[1]), sep = "")
        print(100*"-", "\n", sep = "")
        x += 1


if __name__ == "__main__":
    main()