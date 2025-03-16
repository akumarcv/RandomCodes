def valid_word_abbreviation(word, abbr):
    c1 = 0
    c2 = 0
    while c1 < len(abbr):
        if abbr[c1].isdigit():
            if abbr[c1]=="0":
                return False
            count = []
            while  c1<len(abbr) and abbr[c1].isdigit():
                count.append(abbr[c1])
                c1 = c1 + 1 
            count = int("".join(count))
            c2 = c2+count 
        else:
            if c2>=len(word) or word[c2]!=abbr[c1]:
                return False
            c1 = c1+1
            c2 = c2+1

    return c1 == len(abbr) and c2 == len(word)          

if __name__=="__main__":
    print(valid_word_abbreviation("helloworld" , "4orworld"))