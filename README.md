# XSOR Project - Task 2: Trending Asset into Mean Reverting Asset

This repository contains the final project for Module A of the course "Laboratory of Computational Physics" at UniPD (Master's Degree in Physics of Data), carried out in collaboration with XSOR CAPITAL-Investment Fund.

## Overview
Trending assets are typically non-stationary, displaying a sustained movement in one direction over time, which poses challenges for standard financial analyses. The primary goal of this project is to design a method that removes the trending component of an asset's time series to reveal its mean-reverting properties. 

The project focuses on:
* **Trend Detection:** Applying statistical tests to classify intervals as trending or non-trending.
* **Detrending Methods:** Implementing empirical and statistical approaches to remove the trending component.
* **Validation:** Using a developed mean-reversion metric to assess and validate the effectiveness of the detrending methods.

## Methods and Approaches
The analysis involves several key steps and methodologies:
* **Heiken-Ashi Candlesticks:** A modified candlestick charting technique used to smooth out price action and highlight trends more clearly.
* **Signal Filtering:** Preprocessing the data by discarding 3% of the total power (Area Under the Curve) of the power spectrum to remove high-frequency components.
* **Hurst Exponent:** Evaluating the long-term memory of time series data to determine mean-reverting ($H < 0.5$) or trending ($H > 0.5$) behavior. This was also explored to find an "optimal" interval for rolling means.
* **Mann-Kendall Test:** Applying a non-parametric hypothesis test to assess monotonic trends in the data, accounting for autocorrelation using the `pyMannKendall` package.
* **Autocorrelation Function:** Exploring an algorithm based on the autocorrelation function.
* **Forecasting Metrics:** Evaluating models using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

