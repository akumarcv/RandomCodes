def max_profit(prices: list[int]) -> int:
    """
    Calculate maximum profit possible from buying and selling stock once.
    Uses one-pass approach to track minimum buying price and maximum profit.
    
    Args:
        prices: List of daily stock prices
        
    Returns:
        int: Maximum profit that can be achieved
        
    Time Complexity: O(n) where n is number of prices
    Space Complexity: O(1) as we only use constant extra space
    
    Example:
        >>> max_profit([7,1,5,3,6,4])
        5  # Buy on day 2 (price = 1) and sell on day 5 (price = 6)
    """
    if not prices:
        return 0
        
    buying_day = 0      # Track day with minimum price seen so far
    max_profit = 0      # Track maximum profit possible
    
    for i in range(1, len(prices)):
        # Calculate profit if we sell today
        current_profit = prices[i] - prices[buying_day]
        
        # Update maximum profit if current profit is larger
        if current_profit > max_profit:
            max_profit = current_profit
            
        # Update buying day if we find a lower price
        if prices[i] < prices[buying_day]:
            buying_day = i
            
    return max_profit

def main():
    """
    Driver code to test stock trading algorithm with various price sequences.
    Tests different scenarios including:
    - Regular cases with profit
    - Decreasing prices (no profit)
    - Increasing prices (maximum profit)
    - Complex price movements
    """
    prices = [
        [7, 1, 5, 3, 6, 4],        # Regular case
        [7, 6, 4, 3, 1],           # Decreasing prices
        [1, 2, 3, 4, 5],           # Increasing prices
        [7, 1, 5, 3, 6, 4, 9],     # Multiple peaks
        [1, 2, 3, 4, 5, 6],        # Strictly increasing
        [1, 2, 3, 4, 5, 6, 7],     # Longest increasing
        [7, 6, 5, 4, 3, 2, 1],     # Strictly decreasing
    ]

    for i, price in enumerate(prices, 1):
        profit = max_profit(price)
        print(f"{i}. Input array: {price}")
        print(f"\tMaximum profit that can be achieved: {profit}")
        if profit > 0:
            # Find buying and selling days for visualization
            min_price = min(price)
            max_price = max(price[price.index(min_price):])
            buy_day = price.index(min_price) + 1
            sell_day = price.index(max_price) + 1
            print(f"\tBuy on day {buy_day} at price {min_price}")
            print(f"\tSell on day {sell_day} at price {max_price}")
        else:
            print("\tNo profit possible!")
        print("-" * 100)

if __name__ == "__main__":
    main()