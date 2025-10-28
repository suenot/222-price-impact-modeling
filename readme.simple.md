# Price Impact Modeling -- Explained Simply!

## The Big Splash

Imagine you go to a lemonade stand where there are only 10 cups of lemonade left. If you buy 1 cup, no big deal -- the price stays the same. But what if you try to buy ALL 10 cups at once? The kid running the stand thinks: "Whoa, this person REALLY wants lemonade! I should charge more!"

That's **price impact**: when your trade is so big that it actually moves the price against you.

## The Square Root Rule

Here's a cool pattern that scientists discovered. Let's say buying 1% of all the daily lemonade costs you 1 penny extra. How much extra would buying 4% cost?

You might think 4 pennies (4 times more). But actually, it's only about 2 pennies! That's because price impact follows a **square root rule**:

- 1% of daily volume -> 1x impact
- 4% of daily volume -> 2x impact (square root of 4)
- 9% of daily volume -> 3x impact (square root of 9)
- 16% of daily volume -> 4x impact (square root of 16)

This is actually great news for big traders! If impact were linear, trading 10x more would cost 10x more. But with the square root rule, it only costs about 3.2x more.

## The Puzzle of How Fast to Trade

Imagine you have 100 boxes of toys to sell, but you want to sell them over a whole week instead of all at once. You face a dilemma:

**Sell fast** (dump everything on Monday):
- Good: You know exactly what price you'll get -- no surprises!
- Bad: You'll flood the market and get terrible prices

**Sell slow** (a few boxes each day):
- Good: Each sale barely moves the price
- Bad: The price might drop during the week while you're waiting!

This is the **Almgren-Chriss problem**: finding the perfect selling speed that balances these two worries.

### The Scared Seller vs. The Patient Seller

- A **scared seller** (high risk aversion) sells most on Day 1 because they're terrified the price will crash during the week
- A **patient seller** (low risk aversion) sells the same amount each day, accepting the risk for lower impact

The math finds the exact perfect schedule for any level of scaredness!

## Teaching a Computer to Predict Impact

Instead of using a simple formula, we can train a computer to learn patterns:

**What the computer looks at:**
- How big is the trade compared to normal daily volume?
- How wide is the gap between buy and sell prices? (the "spread")
- How jumpy is the price today? (volatility)
- Is this a buy or a sell?

**What it learns:**
- "When the spread is wide AND the trade is big, impact is HUGE"
- "Morning trades have less impact than lunchtime trades"
- "When Bitcoin is calm, even big trades don't move the price much"

It's like teaching a friend to estimate how long it takes to drive somewhere. At first they just use distance, but eventually they learn about traffic, weather, time of day, and road conditions too!

## The Report Card: Transaction Cost Analysis

After you make trades, you want to know: "Did I do a good job?" That's what **Transaction Cost Analysis (TCA)** does. It's like a report card for your trading:

- **Implementation Shortfall**: "The price was $100 when I decided to buy, but I actually paid $100.05. I lost 5 cents per share!" (This is like noticing the price went up while you were walking to the store.)

- **VWAP Slippage**: "Most people paid an average of $100.03 today, but I paid $100.05. I did slightly worse than average." (This is like comparing your test score to the class average.)

- **Spread Cost**: "There was a 2-cent gap between the buy price and sell price, so it cost me 1 cent just to cross that gap." (This is like the fee the lemonade stand charges.)

## A Real Trading Example

Let's say you want to buy $50,000 worth of Bitcoin:

1. **Square root model says**: "Based on today's volume and volatility, this trade will move the price about 2 basis points (0.02%)"
2. **Almgren-Chriss says**: "Split it into 10 pieces over the next 2 hours"
3. **ML model says**: "Actually, right now the order book is thin, so expect 3 basis points"
4. **After trading, TCA says**: "You paid 2.5 basis points of slippage -- not bad!"

## Why This Matters

| Without impact modeling | With impact modeling |
|---|---|
| "My strategy makes 10% per year!" | "After impact costs, it makes 6% per year" |
| Buy everything NOW! | Split into optimal-sized pieces |
| Trade the same way every time | Adjust speed based on current conditions |
| Backtest looks amazing | Backtest is realistic |

## Summary for Quick Learners

1. **Price impact** = Your trade pushes the price against you. Bigger trades cause bigger impact.
2. **Square root law** = Doubling your trade size only increases impact by about 41%, not 100%.
3. **Almgren-Chriss** = Math that tells you the perfect speed to trade, balancing "sell fast" vs. "sell cheap."
4. **ML prediction** = Teaching a computer to predict impact from many signals at once.
5. **TCA** = A report card that tells you how well you actually traded.

Think of it like this: if the market is a swimming pool, the square root law tells you how big a splash you'll make, Almgren-Chriss tells you how slowly to step in, and ML watches the water to predict your splash even better!
