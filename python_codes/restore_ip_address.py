def helper(result, s, prefix):
    if prefix.count(".") > 3 or not s:
        return
    if prefix.count(".") == 3 and len(s) <= 3 and int(s) <= 255 and str(int(s)) == s:
        result.append(prefix + s)
        return

    if len(s) > 0:
        helper(result, s[1:], prefix + s[0] + ".")
    if len(s) > 1 and str(int(s[:2])) == s[:2]:
        helper(result, s[2:], prefix + s[0:2] + ".")
    if len(s) > 3 and int(s[:3]) <= 255 and str(int(s[:3])) == s[:3]:
        helper(result, s[3:], prefix + s[0:3] + ".")


def restore_ip_addresses(s):
    if not s or len(s) > 12 or len(s) < 4:
        return []
    result = []
    helper(result, s, "")
    # Replace this placeholder return statement with your code
    return result


def main():
    ip_addresses = [
        "0000",
        "25525511135",
        "12121212",
        "113242124",
        "199219239",
        "121212",
        "25525511335",
    ]

    for i in range(len(ip_addresses)):
        print(i + 1, ".\t Input addresses: '", ip_addresses[i], "'", sep="")
        print(
            "\t Possible valid IP Addresses are: ",
            restore_ip_addresses(ip_addresses[i]),
            sep="",
        )
        print("-" * 100)


if __name__ == "__main__":
    main()
