def helper(result, s, prefix):
    """
    Recursive backtracking helper to generate valid IP address combinations.
    Explores possible segment divisions and validates each segment.
    
    Args:
        result: List to store valid IP addresses found
        s: Remaining string to process
        prefix: Current partial IP address being built
        
    Returns:
        None: Updates result list in-place
        
    Time Complexity: O(1) as IP addresses have fixed constraints
    Space Complexity: O(1) for recursion stack (max depth is 4)
    """
    # Base cases for termination
    if prefix.count(".") > 3 or not s:
        return  # Too many dots or no characters left
    if prefix.count(".") == 3 and len(s) <= 3 and int(s) <= 255 and str(int(s)) == s:
        result.append(prefix + s)  # Valid IP found, add to results
        return

    # Try different segment lengths (1, 2, or 3 digits)
    if len(s) > 0:
        # Try 1-digit segment
        helper(result, s[1:], prefix + s[0] + ".")
    if len(s) > 1 and str(int(s[:2])) == s[:2]:
        # Try 2-digit segment (avoid leading zeros)
        helper(result, s[2:], prefix + s[0:2] + ".")
    if len(s) > 3 and int(s[:3]) <= 255 and str(int(s[:3])) == s[:3]:
        # Try 3-digit segment (must be ≤ 255, avoid leading zeros)
        helper(result, s[3:], prefix + s[0:3] + ".")


def restore_ip_addresses(s):
    """
    Restore all possible valid IP addresses from a string of digits.
    Valid IP address has format A.B.C.D where A,B,C,D are numbers from 0-255.
    
    Args:
        s: Input string of digits
        
    Returns:
        list: All possible valid IP addresses that can be formed
        
    Time Complexity: O(1) as IP addresses have fixed constraints
    Space Complexity: O(1) as there can be at most 27 valid IP addresses
    
    Example:
        >>> restore_ip_addresses("25525511135")
        ["255.255.11.135", "255.255.111.35"]
    """
    if not s or len(s) > 12 or len(s) < 4:
        return []  # Invalid input: empty, too long, or too short
    result = []    # Store valid IP addresses
    helper(result, s, "")  # Start recursive backtracking
    return result


def main():
    """
    Driver code to test IP address restoration with multiple examples.
    Tests various input strings including:
    - All zeros
    - Common length strings
    - Strings with exact segment boundaries
    - Invalid strings (too long/short)
    - Strings with leading zeros
    """
    ip_addresses = [
        "0000",            # Special case: all zeros
        "25525511135",     # Example with multiple valid IPs
        "12121212",        # Evenly distributed digits
        "113242124",       # Mixed digit lengths
        "199219239",       # Values near upper limit (255)
        "121212",          # Shorter string
        "25525511335",     # Contains invalid segment value
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