import sys, os, pdb


class Solution:
    def myAtoi(self, st):
        if st is None:
            return st
        sign = 1
        num = 0
        numberhasstarted = False
        numberdone = False
        count = 0
        for i in st:
            if not numberdone:
                if i == " ":
                    if not numberhasstarted:
                        continue
                    else:
                        numberdone = True
                elif i == "-" or i == "+":
                    if not numberhasstarted:
                        sign = 1 if i == "+" else -1
                        numberhasstarted = True
                    else:
                        numberdone = True
                elif ord(i) >= ord("0") and ord(i) <= ord("9"):
                    num = num * 10 + (ord(i) - 48)
                    numberhasstarted = True
                else:
                    numberdone = True

        if sign * num > ((2**31) - 1):
            return 2**31 - 1
        if sign * num < (-(2**31)):
            return -(2**31)
        return sign * num


obj = Solution()
st = "  +- 1213"
val = obj.myAtoi(st)
print(val)
