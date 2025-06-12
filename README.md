# 🎯 Advanced Options Pricing Simulator

A comprehensive, interactive web application for options pricing using multiple financial models with real-time visualizations and Greek calculations.

## 🌟 Features

### Multiple Pricing Models
- **Black-Scholes Model**: Classic analytical solution for European options
- **Binomial Tree Model**: Discrete-time model suitable for American options
- **Monte Carlo Simulation**: Probabilistic approach with path visualization

### Comprehensive Analytics
- **Real-time Pricing**: Calculate option prices instantly with parameter changes
- **Greeks Calculation**: Complete set of risk sensitivities (Delta, Gamma, Theta, Vega, Rho)
- **Model Comparison**: Side-by-side comparison of different pricing models
- **Interactive Visualizations**: 3D volatility surfaces, Monte Carlo path analysis

### Advanced Visualizations
- **Volatility Surface**: 3D interactive surface showing option prices across different spot prices and volatilities
- **Monte Carlo Analysis**: Price path distributions, payoff histograms, and convergence analysis
- **Greeks Dashboard**: Comparative visualization of risk metrics across models

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.7
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/abhikumar63/advanced-option-pricing-simulator.git
cd advanced-option-pricing-simulator
```

2. Install required packages:
```bash
pip install streamlit numpy pandas plotly matplotlib scipy
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 💡 Usage

### Basic Parameters
- **Current Stock Price**: The current market price of the underlying asset
- **Strike Price**: The exercise price of the option
- **Time to Expiry**: Time remaining until option expiration (in years)
- **Volatility**: Expected volatility of the underlying asset (%)
- **Risk-Free Rate**: Current risk-free interest rate (%)
- **Dividend Yield**: Annual dividend yield of the underlying asset (%)

### Model Selection
Choose from three sophisticated pricing models:

1. **Black-Scholes**: Best for European options with constant parameters
2. **Binomial Tree**: Ideal for American options and dividend-paying stocks
3. **Monte Carlo**: Perfect for complex derivatives and path-dependent options

### Advanced Settings
- **Monte Carlo Simulations**: Number of price paths (1,000 - 1,000,000)
- **Binomial Steps**: Number of time steps in the binomial tree (10 - 500)

## 📊 Model Details

### Black-Scholes Model
The Black-Scholes model provides an analytical solution for European options:

**Formula**: 
- Call: `C = S₀e^(-qT)N(d₁) - Ke^(-rT)N(d₂)`
- Put: `P = Ke^(-rT)N(-d₂) - S₀e^(-qT)N(-d₁)`

**Assumptions**:
- Constant volatility and risk-free rate
- Log-normal price distribution
- No transaction costs
- Continuous trading

### Binomial Tree Model
A discrete-time model that models price movements as up/down moves:

**Advantages**:
- Handles American-style options
- Accommodates dividend payments
- Flexible time structure

**Parameters**:
- Up factor: `u = e^(σ√Δt)`
- Down factor: `d = 1/u`
- Risk-neutral probability: `p = (e^((r-q)Δt) - d)/(u - d)`

### Monte Carlo Simulation
Uses random sampling to simulate thousands of possible price paths:

**Process**:
1. Generate random price paths using geometric Brownian motion
2. Calculate payoffs for each path
3. Discount back to present value
4. Average across all simulations

**Advantages**:
- Handles complex payoff structures
- Provides distribution insights
- Easily extensible to exotic options

## 🔢 Greeks Explained

The application calculates all major Greeks for comprehensive risk analysis:

- **Delta (Δ)**: Price sensitivity to underlying asset price changes
- **Gamma (Γ)**: Rate of change of Delta
- **Theta (Θ)**: Time decay of option value
- **Vega (ν)**: Sensitivity to volatility changes
- **Rho (ρ)**: Sensitivity to interest rate changes

## 📈 Visualizations

### Volatility Surface
Interactive 3D surface showing how option prices vary with:
- Underlying asset price (X-axis)
- Volatility levels (Y-axis)
- Option price (Z-axis)

### Monte Carlo Analysis
Comprehensive visualization including:
- **Price Paths**: Sample of simulated asset price trajectories
- **Final Price Distribution**: Histogram of terminal asset prices
- **Payoff Distribution**: Distribution of option payoffs
- **Convergence Plot**: Shows how the estimate converges with more simulations

## 🎨 User Interface

### Modern Design Elements
- **Gradient Cards**: Beautiful metric displays
- **Interactive Controls**: Real-time parameter adjustment
- **Progress Indicators**: Visual feedback during calculations
- **Responsive Layout**: Works on desktop and mobile devices

### Navigation
- **Sidebar**: All input parameters and model selection
- **Main Panel**: Results, visualizations, and analysis
- **Expandable Sections**: Educational content and advanced settings

## 🔧 Technical Implementation

### Core Technologies
- **Streamlit**: Web application framework
- **NumPy**: Numerical computations
- **SciPy**: Statistical functions and distributions
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation and analysis

### Architecture
```
├── OptionsPricingSimulator (Class)
│   ├── black_scholes()      # Analytical pricing
│   ├── binomial_tree()      # Tree-based pricing
│   └── monte_carlo()        # Simulation-based pricing
├── Visualization Functions
│   ├── create_price_surface()    # 3D volatility surface
│   ├── create_monte_carlo_viz()  # MC analysis plots
│   └── create_greeks_chart()     # Greeks comparison
└── Streamlit UI Components
    ├── Parameter inputs
    ├── Model selection
    └── Results display
```

## 📚 Educational Content

The application includes built-in educational materials explaining:
- Model assumptions and limitations
- When to use each pricing model
- Interpretation of Greeks
- Market risk factors

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Black-Scholes-Merton model for foundational options pricing theory
- Streamlit team for the excellent web app framework
- Plotly for interactive visualization capabilities
- The quantitative finance community for continuous innovation

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the educational content in the app
- Review the model assumptions and limitations

## 🔮 Future Enhancements

- [ ] Asian options pricing
- [ ] Barrier options support
- [ ] Real-time market data integration
- [ ] Portfolio risk analysis
- [ ] Implied volatility calculations
- [ ] Options strategy analyzer
- [ ] Historical volatility estimation
- [ ] Sensitivity analysis tools

---

**Built with ❤️ for the quantitative finance community**
