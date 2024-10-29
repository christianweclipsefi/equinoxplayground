import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from typing import List, Tuple

class StableSwapSimulator:
    def __init__(self):
        # Constants
        self.a = 10
        self.b = self.a ** (1 / self.a)
        self.max_loop_limit = 1000
        self.A_PRECISION = 100
        self.max_A = 10 ** 6
        self.MAX_POOL_VALUE = 25000000  # Increased max per token
        
        # Initial pool values
        self.pool_values = [1000000, 1000000]  # Initial pools A and B
        self.n_tokens = len(self.pool_values)
        self.__A = 85
        self.A = self.__A * self.A_PRECISION
        self.xp = self.pool_values.copy()
        
    def multiply_pool_values(self, v=None) -> float:
        if v is None:
            v = self.pool_values
        mul = 1
        for val in v:
            mul *= val
        return mul
    
    def sum_pool_values(self) -> float:
        return sum(self.pool_values)
    
    def sum_max_pool_values(self) -> float:
        return self.MAX_POOL_VALUE * 2  # Increased total max
        
    def get_d(self) -> float:
        s = sum(self.xp)
        if s == 0:
            return 0
            
        d = s
        n_a = self.A * self.n_tokens
        
        for _ in range(self.max_loop_limit):
            d_p = d
            for x in self.xp:
                d_p = (d_p * d) / (x * self.n_tokens)
            prev_d = d
            d = (
                ((n_a * s / self.A_PRECISION) + (d_p * self.n_tokens)) * d
            ) / (
                ((n_a - self.A_PRECISION) * d / self.A_PRECISION) + ((self.n_tokens + 1) * d_p)
            )
            
            if self.within1(d, prev_d):
                return d
                
        raise Exception("D does not converge")
    
    def get_y(self, x: float, token_index_from: int = 0, token_index_to: int = 1) -> float:
        d = self.get_d()
        c = d
        s = 0
        n_a = self.n_tokens * self.A
        
        for i in range(self.n_tokens):
            if i == token_index_from:
                _x = x
            elif i != token_index_to:
                _x = self.xp[i]
            else:
                continue
            s += _x
            c = (c * d) / (_x * self.n_tokens)
            
        c = ((c * d) * self.A_PRECISION) / (n_a * self.n_tokens)
        b = s + ((d * self.A_PRECISION) / n_a)
        y = d
        
        for _ in range(self.max_loop_limit):
            y_prev = y
            y = ((y * y) + c) / (((y * 2) + b) - d)
            if self.within1(y, y_prev):
                return y
                
        raise Exception("Approximation did not converge")
    
    def calculate_swap(self, dx: float, token_index_from: int = 0, token_index_to: int = 1) -> Tuple[float, float]:
        fee = 0  # Can be adjusted, e.g., 0.0004
        
        x = dx + self.xp[token_index_from]
        y = self.get_y(x, token_index_from, token_index_to)
        dy = self.xp[token_index_to] - y
        dy_fee = dy * fee if fee else 0
        dy = max(dy - dy_fee, 0)
        return dy, dy_fee
    
    @staticmethod
    def within1(a: float, b: float) -> bool:
        return abs(a - b) <= 1
    
    def update_a_coeff(self, new_a: float):
        self.__A = new_a
        self.A = self.__A * self.A_PRECISION
        
    def update_pool_values(self, index: int, value: float):
        self.pool_values[index] = value
        self.xp = self.pool_values.copy()
    
    def get_a(self) -> float:
        return self.__A

def main():
    st.title("StableSwap Simulator")
    
    simulator = StableSwapSimulator()
    
    # Pool configuration
    st.header("Pool Configuration")
    
    # Token sliders with increased max_value
    cols = st.columns(len(simulator.pool_values))
    for i, col in enumerate(cols):
        with col:
            value = st.number_input(
                f"Token {chr(65 + i)} Pool",
                min_value=0.0,
                max_value=25000000.0,  # Increased from 2000000.0 to 25000000.0
                value=float(simulator.pool_values[i]),
                step=1000.0
            )
            simulator.update_pool_values(i, value)
    
    # A coefficient slider
    a_coeff = st.slider(
        "A Coefficient",
        min_value=1,
        max_value=1000,
        value=simulator.get_a(),
        step=1
    )
    simulator.update_a_coeff(a_coeff)
    
    # Swap simulation
    st.header("Swap Simulation")
    
    col1, col2 = st.columns(2)
    with col1:
        token_in = st.selectbox(
            "Token In",
            options=[f"Token {chr(65 + i)}" for i in range(len(simulator.pool_values))]
        )
    with col2:
        token_out = st.selectbox(
            "Token Out",
            options=[f"Token {chr(65 + i)}" for i in range(len(simulator.pool_values))],
            index=1
        )
    
    amount_in = st.number_input(
        "Amount to swap",
        min_value=0.0,
        max_value=25000000.0,  # Increased from 1000000.0 to 25000000.0
        value=0.0,
        step=1000.0
    )
    
    if amount_in > 0:
        token_in_idx = ord(token_in[-1]) - 65
        token_out_idx = ord(token_out[-1]) - 65
        amount_out, fee = simulator.calculate_swap(amount_in, token_in_idx, token_out_idx)
        
        st.write(f"Amount out: {amount_out:.2f}")
        st.write(f"Price: {(amount_out/amount_in if amount_in else 0):.5f}")
        
        # Plot the curve
        x = np.linspace(0, simulator.sum_max_pool_values(), 1000)
        y = [simulator.calculate_swap(float(x_val))[0] for x_val in x]
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel("Amount In")
        ax.set_ylabel("Amount Out")
        ax.set_title("StableSwap Curve")
        ax.grid(True)
        
        if amount_in > 0:
            ax.scatter([amount_in], [amount_out], color='red')
            
        st.pyplot(fig)

if __name__ == "__main__":
    main()