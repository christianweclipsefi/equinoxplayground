import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Global variables
width = 800
height = 500
range_values = {
    'k': 4,
    's': 0,
    's2': 1
}

# Function to render charts
def render_chart(title, formula, derivative=None, points=None, domain=(0, 10)):
    x = np.linspace(domain[0], domain[1], 400)
    y = eval(formula)

    plt.figure(figsize=(width / 100, height / 100))
    plt.plot(x, y, label=title, color='blue')

    if derivative:
        dydx = eval(derivative)
        plt.plot(x, dydx, label=f"{title} Derivative", linestyle='--', color='orange')

    if points:
        for i, point_set in enumerate(points):
            px, py = zip(*point_set)
            plt.plot(px, py, label=f"Points {i+1}", linestyle='-', marker='o')

    plt.title(title)
    plt.xlabel('X ($)')
    plt.ylabel('Y ($)')
    plt.xlim(domain)
    plt.ylim(domain)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# Function to draw charts
def draw():
    formulas = {
        'constantSum': 'range_values["k"] - x',
        'constantProduct': 'range_values["k"] / x',
        'stableswap': '(1 + x)**-1 * (range_values["k"] + (range_values["k"]**2) * (2**-2) - x)',
        'chi': '(range_values["s"] * range_values["k"] + ((range_values["k"] / 2)**2) - range_values["s"] * x) * ((range_values["s"] + x)**-1)'
    }

    derivatives = {
        'constantSum': '-1',
        'constantProduct': '-range_values["k"] / x**2',
        'stableswap': '(-(1 + x)**-1) * (1 + (range_values["k"] + (range_values["k"]**2) * (2**-2) - x) * ((1 + x)**-1))',
        'chi': '(-(range_values["s"] + x)**-1) * (range_values["s"] + (range_values["s"] * range_values["k"] + (range_values["k"]**2) * (2**-2) - (range_values["s"] * x)) * ((range_values["s"] + x)**-1))'
    }

    render_chart(
        title='x + y = C',
        formula=formulas['constantSum'],
        domain=(0, 10)
    )

    render_chart(
        title='x * y = k',
        formula=formulas['constantProduct'],
        derivative=derivatives['constantProduct'],
        domain=(0, 10)
    )

    render_chart(
        title='x + y + x * y = D + (D/n)^n',
        formula=formulas['stableswap'],
        derivative=derivatives['stableswap'],
        domain=(0, 10)
    )

    render_chart(
        title='𝛘(x+y+xy)=C',
        formula=formulas['chi'],
        derivative=derivatives['chi'],
        domain=(0, 10)
    )

# Streamlit app
def main():
    st.title("Chart Simulator")

    # Bind inputs
    range_values['k'] = st.slider('k', 1, 10, 4)
    range_values['s'] = st.slider('s', 0, 10, 0)
    range_values['s2'] = st.slider('s2', 0, 10, 1)

    draw()

if __name__ == "__main__":
    main()
