import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
from typing import List

# Set precision for decimal calculations
getcontext().prec = 30

class StableSwapSimulator:
    def __init__(self, amp: int, initial_balances: List[Decimal]):
        """
        Initialize StableSwap pool simulator
        Based on Astroport's implementation
        
        Args:
            amp: Amplification coefficient (A)
            initial_balances: Initial token balances [xASTRO, eclipASTRO]
        """
        self.amp = amp
        self.n_coins = 2
        self.balances = initial_balances
        self.precision = Decimal('1e-6')
        
    def _compute_d(self) -> Decimal:
        """
        Compute the stableswap invariant (D)
        Using Newton's method to solve for D
        """
        sum_x = sum(self.balances)
        if sum_x == 0:
            return Decimal(0)
            
        d = sum_x
        leverage = Decimal(self.amp) * Decimal(self.n_coins)
        
        for _ in range(64):  # Max iterations from Astroport
            d_prev = d
            
            # Calculate D_P = D^(n+1) / (n^n * prod(x_i))
            d_p = d ** 3 / (self.balances[0] * self.balances[1] * Decimal(4))
            
            # Apply Newton's method formula
            d = (leverage * sum_x + d_p * Decimal(2)) * d / ((leverage - Decimal(1)) * d + Decimal(3) * d_p)
            
            # Check convergence
            if abs(d - d_prev) <= self.precision:
                return d
                
        raise Exception("D calculation failed to converge")

    def calculate_price_impact(self, dx: Decimal) -> Decimal:
        """
        Calculate price impact for selling dx amount of token 0
        Returns price impact as a percentage
        """
        if dx == 0:
            return Decimal(0)
            
        # Calculate initial price
        d = self._compute_d()
        x0 = self.balances[0]
        y = self.balances[1]
        
        # Calculate new balance after swap
        x1 = x0 + dx
        new_balances = [x1, y]
        
        # Calculate new D
        old_d = d
        self.balances = new_balances
        new_d = self._compute_d()
        
        # Restore original balances
        self.balances = [x0, y]
        
        # Calculate price impact
        price_impact = abs((new_d - old_d) / old_d) * Decimal(100)
        return price_impact

def main():
    st.title("Astroport StableSwap Pool Simulator")
    
    # Pool configuration
    st.header("Pool Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        xastro_amount = st.number_input(
            "Initial xASTRO amount",
            min_value=1,
            value=375000,
            step=1000
        )
        
    with col2:
        eclip_amount = st.number_input(
            "Initial eclipASTRO amount",
            min_value=1,
            value=375000,
            step=1000
        )
    
    # Amplification parameter
    amp = st.slider(
        "Amplification Parameter (A)",
        min_value=1,
        max_value=1000,
        value=100,
        step=1
    )
    
    # Create simulator instance
    simulator = StableSwapSimulator(
        amp=amp,
        initial_balances=[Decimal(xastro_amount), Decimal(eclip_amount)]
    )
    
    # Trade simulation
    st.header("Trade Simulation")
    trade_amount = st.number_input(
        "Amount of xASTRO to sell",
        min_value=0,
        value=100000,
        step=1000
    )
    
    # Calculate and plot price impact for range of trades
    trade_range = np.linspace(0, trade_amount * 2, 100)
    price_impacts = [float(simulator.calculate_price_impact(Decimal(x))) for x in trade_range]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trade_range, price_impacts)
    ax.set_xlabel("Trade Size (xASTRO)")
    ax.set_ylabel("Price Impact (%)")
    ax.set_title("Price Impact vs Trade Size")
    ax.grid(True)
    
    # Highlight selected trade amount
    if trade_amount > 0:
        price_impact = simulator.calculate_price_impact(Decimal(trade_amount))
        ax.scatter([trade_amount], [float(price_impact)], color='red', s=100)
        st.write(f"Price Impact for {trade_amount:,} xASTRO: {price_impact:.2f}%")
    
    st.pyplot(fig)

if __name__ == "__main__":
    main()