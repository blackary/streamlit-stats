import streamlit as st
from random import random
import pandas as pd
from matplotlib import pyplot as plt

"# What is the mean of a set of numbers?"

nums = st.slider("How many numbers", min_value=1, max_value=20, value=10)
min_value = st.slider("Minimimum Value", min_value=0, max_value=100, value=0)
max_value = st.slider("Maximum Value", min_value=min_value, max_value=100, value=100)


def mean(items: list[float]) -> float:
    return sum(items) / len(items)


def get_values(num: int) -> list[float]:
    values: list[float] = []
    for num in range(nums):
        scale = random()
        value = min_value + scale * (max_value - min_value)
        values.append(round(value, 1))
    return values


values = get_values(nums)

if st.button("regenerate"):
    values = get_values(nums)

df = pd.DataFrame({"values": sorted(values)})

"## Values"
st.text(values)

mean_val = mean(values)

"## Mean"
mean_val

fig, ax = plt.subplots()

df.plot.bar(ax=ax)

plt.axhline(y=mean_val, color="r", label=f"mean: {mean_val:0.1f}", linestyle="dashed")

plt.legend()

st.pyplot(fig)
