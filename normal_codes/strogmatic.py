def is_strobogrammatic(num):
    mapping = {"8": "8", "0": "0", "6": "9", "9": "6", "1": "1"}
    left = 0
    right = len(num)-1
    while left<=right:
      if num[left] not in mapping.keys() or num[right] not in mapping.keys():
        return False
      if mapping[num[left]]==num[right]:
        left = left + 1
        right = right -1 
      else: 
        return False
    
    # Replace this placeholder return statement with your code
    return True

# Driver code
def main():
    nums = [
        "609",   
        "88",   
        "962",  
        "101",  
        "123", 
        "619"
    ]

    i = 0
    for num in nums:
        print(i + 1, ".\tnum: ", num, sep="")
        print("\n\tIs strobogrammatic: ", is_strobogrammatic(num), sep="")
        print("-" * 100)
        i += 1

if __name__ == "__main__":
    main()