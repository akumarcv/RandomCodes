def longest_palindromic_substrings(s):

    if not s:
        return 0
    dp = [[False for _ in range(len(s))] for _ in range(len(s))]
    
    for i in range(len(s)):
        dp[i][i] = True
    
    max_len = 1

    for i in range(len(s)-1):
        if s[i]==s[i+1]:
            dp[i][i+1] = True
            max_len = max(max_len, 2)
    
    for i in range(3, len(s)+1):
        k = 0
        for j in range(i-1 , len(s)):
            dp[k][j] = dp[k+1][j-1] and (s[k]==s[j])
            if dp[k][j]:
                max_len = max(max_len, j-k+1)
            k+=1
        
    return max_len


# Driver code
def main():
    strings = ['cat', 'lever', 'xyxxyz', 'wwwwwwwwww', 'tattarrattat']
    
    for i in range(len(strings)):
        print(i + 1, ".\t Input string: '", strings[i], "'", sep="")
        result = longest_palindromic_substrings(strings[i])
        print("\t Number of palindromic substrings: ", result, sep="")
        print("-" * 100)
    
if __name__ == '__main__':
    main()