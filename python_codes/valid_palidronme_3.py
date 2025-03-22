def is_palindrome(strings):

    # Replace this placeholder return statement with your code
    left = 0
    right = len(strings) - 1
    mismatch = 0
    if len(strings) == 1:
        return True
    while left < right:
        if strings[left] == strings[right]:
            left += 1
            right -= 1
        else:
            if strings[left + 1] == strings[right]:
                i = left + 1
                length = right - left
                mismatch = 1
                for k in range(length // 2):
                    if strings[i + k] != strings[right - k]:
                        mismatch += 1
                print(f"if {mismatch}")
                if mismatch == 1:
                    return True
            else:
                mismatch = 1
                i = right - 1
                length = right - left
                for k in range(length // 2):
                    if strings[left + k] != strings[i - k]:
                        mismatch += 1
                print(f"else {mismatch}")
                if mismatch == 1:
                    return True

            left += 1
            right -= 1
    if mismatch == 0:
        return True
    else:
        return False


if __name__ == "__main__":
    print(
        is_palindrome(
            "ElDXxFgmiPzvjUmBcpjyMYYtzcuBEmgWwvkFePovorAcBXbuArdvpwSpGlXExWumEiifqcDflfzMOPvNmrpPoUGqZCOfrBNeSevHolDgiiHhpTUgaJkcCmLZPKoqwfOqmXSXCRdkJLLGKCXCKIOjssRrsyUusKnmGZLKqteAMziPZgsigZmDciZFAzOcTkvPBrbBKnALPrxpYQEnHhTZdVGAZgjfMmzTdqbicrZGhUgerDGMNXEPEhRCwXRukJeljZYwwVlxffdPrWROMnTmRqfObVECBjIewuAJvdAiymxhxbGeBhWpIMhtTpZRFYenIUqmldlDDESzHuoXuxBHGasGhXpkukYUNgmUxGAPzNdlHeiGdRgCaLBBuqeiNvTyByDPCEzLpOtvMsKmMvmxwivNSOjVcVunRNgOmuNvESYBAjfWeZCVsVVscRnzMAAQeAYjgtYpkDdhgQLqgLplduOhVkaDtNtiRKKLivFFWKCPGLxryjNkkNjyrxLGPCKWFFviLKKRitNtDakVhOudlpLgqLQghdDkpYtgjYAeQAAMznRcsVVsVCZeWfjABYSEvNumOgNRnuVcVjOSNviwxmvMmKsMvtOpLzECPDyByTvNiequBBLaCgRdGieHldNzPAGxUmgNUYkukpXhGsaGHBxuXouHzSEDDldlmqUIneYFRZpTthMIpWhBeGbxhxmyiAdvJAuweIjBCEVbOfqRmTnMORWrPdffxlVwwYZjleJkuRXwCRhEPEXNMGDregUhGZrcibqdTzmMfjgZAGVdZThHnEQYpxrPLAnKBbrBPvkTcOzAFZicDmZgisgZPizMAetqKLZGmnKsuUysrRssjOIKCXCKGLLJkdRCXSXmqOfwqoKPZLmCckJagUTphHiigDloHveSeNBrfOCZqGUoPprmNvPOMzflfDcqfiiEmuWxEXlGpSwpvdrAubXBcArovoPeFkvwWgmEBucztYYMyjpcBmUjvzPimgFxXDlE"
        )
    )
