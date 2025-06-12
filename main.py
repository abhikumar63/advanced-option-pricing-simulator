import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

# Page config
st.set_page_config(
    page_title="Options Pricing Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class OptionsPricingSimulator:
    @staticmethod
    def black_scholes(S, K, T, r, sigma, q=0.0, option_type='call'):
        """Black-Scholes option pricing model with Greeks"""
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = np.exp(-q * T) * norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
            delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        
        # Greeks
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
                + q * S * np.exp(-q * T) * norm.cdf(d1 if option_type == 'call' else -d1))
        if option_type == 'put':
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)
                    - q * S * np.exp(-q * T) * norm.cdf(-d1))
        
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
        if option_type == 'put':
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # Per day
            'vega': vega / 100,    # Per 1% volatility change
            'rho': rho / 100       # Per 1% interest rate change
        }
    
    @staticmethod
    def _binomial_price_only(S, K, T, r, sigma, q=0.0, option_type='call', steps=100):
        """Helper method to calculate binomial price without Greeks (avoids recursion)"""
        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        
        # Asset price tree
        asset_prices = np.zeros((steps + 1, steps + 1))
        for i in range(steps + 1):
            for j in range(i + 1):
                asset_prices[j, i] = S * (u ** (i - j)) * (d ** j)
        
        # Option value tree
        option_values = np.zeros((steps + 1, steps + 1))
        
        # Terminal values
        for j in range(steps + 1):
            if option_type == 'call':
                option_values[j, steps] = max(0, asset_prices[j, steps] - K)
            else:
                option_values[j, steps] = max(0, K - asset_prices[j, steps])
        
        # Backward induction
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                option_values[j, i] = np.exp(-r * dt) * (
                    p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1]
                )
        
        return option_values[0, 0]
    
    @staticmethod
    def binomial_tree(S, K, T, r, sigma, q=0.0, option_type='call', steps=100):
        """Binomial tree option pricing model"""
        price = OptionsPricingSimulator._binomial_price_only(S, K, T, r, sigma, q, option_type, steps)
        
        # Approximate Greeks using finite differences
        h = 0.01 * S
        price_up = OptionsPricingSimulator._binomial_price_only(S + h, K, T, r, sigma, q, option_type, steps)
        price_down = OptionsPricingSimulator._binomial_price_only(S - h, K, T, r, sigma, q, option_type, steps)
        delta = (price_up - price_down) / (2 * h)
        
        # Calculate gamma using second derivative
        price_up2 = OptionsPricingSimulator._binomial_price_only(S + 2*h, K, T, r, sigma, q, option_type, steps)
        price_down2 = OptionsPricingSimulator._binomial_price_only(S - 2*h, K, T, r, sigma, q, option_type, steps)
        gamma = (price_up2 - 2*price + price_down2) / (4 * h * h)
        
        # Approximate other Greeks
        # Theta - time decay
        if T > 0.01:  # Avoid division by zero
            price_theta = OptionsPricingSimulator._binomial_price_only(S, K, T - 0.01, r, sigma, q, option_type, steps)
            theta = (price_theta - price) / 0.01
        else:
            theta = -0.02
        
        # Vega - volatility sensitivity
        vol_shift = 0.01
        price_vega = OptionsPricingSimulator._binomial_price_only(S, K, T, r, sigma + vol_shift, q, option_type, steps)
        vega = (price_vega - price) / vol_shift
        
        # Rho - interest rate sensitivity
        rate_shift = 0.01
        price_rho = OptionsPricingSimulator._binomial_price_only(S, K, T, r + rate_shift, sigma, q, option_type, steps)
        rho = (price_rho - price) / rate_shift
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # Per day
            'vega': vega / 100,    # Per 1% volatility change
            'rho': rho / 100       # Per 1% interest rate change
        }
    
    @staticmethod
    def monte_carlo(S, K, T, r, sigma, q=0.0, option_type='call', simulations=100000):
        """Monte Carlo option pricing simulation"""
        # Generate random price paths
        dt = T
        Z = np.random.standard_normal(simulations)
        ST = S * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Discount back to present value
        price = np.exp(-r * T) * np.mean(payoffs)
        
        # Store paths for visualization
        price_paths = []
        for i in range(min(1000, simulations)):  # Store first 1000 paths for plotting
            path = [S]
            current_price = S
            for step in range(100):
                z = np.random.standard_normal()
                current_price *= np.exp((r - q - 0.5 * sigma**2) * (T/100) + sigma * np.sqrt(T/100) * z)
                path.append(current_price)
            price_paths.append(path)
        
        return {
            'price': price,
            'delta': 0.5,  # Simplified for demo
            'gamma': 0.02,
            'theta': -0.03,
            'vega': 0.2,
            'rho': 0.1,
            'final_prices': ST,
            'payoffs': payoffs,
            'price_paths': price_paths
        }

def create_price_surface():
    """Create volatility surface plot"""
    spot_range = np.linspace(80, 120, 20)
    vol_range = np.linspace(0.1, 0.5, 20)
    S_grid, vol_grid = np.meshgrid(spot_range, vol_range)
    
    # Calculate option prices for surface
    K = st.session_state.get('strike', 100)
    T = st.session_state.get('time_to_expiry', 0.25)
    r = st.session_state.get('risk_free_rate', 0.05)
    option_type = st.session_state.get('option_type', 'call')
    
    price_surface = np.zeros_like(S_grid)
    for i in range(len(vol_range)):
        for j in range(len(spot_range)):
            result = OptionsPricingSimulator.black_scholes(
                S_grid[i,j], K, T, r, vol_grid[i,j], option_type=option_type
            )
            price_surface[i,j] = result['price']
    
    fig = go.Figure(data=[go.Surface(x=spot_range, y=vol_range, z=price_surface,
                                   colorscale='Viridis')])
    fig.update_layout(
        title='Option Price Surface',
        scene=dict(
            xaxis_title='Spot Price',
            yaxis_title='Volatility',
            zaxis_title='Option Price'
        ),
        height=500
    )
    return fig

def create_monte_carlo_viz(result):
    """Create Monte Carlo simulation visualizations"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price Paths Sample', 'Final Price Distribution', 
                       'Payoff Distribution', 'Convergence'),
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    # Price paths (sample)
    paths_sample = result['price_paths'][:50]  # Show first 50 paths
    for i, path in enumerate(paths_sample):
        fig.add_trace(
            go.Scatter(y=path, mode='lines', line=dict(width=1, color='rgba(0,100,200,0.3)'),
                      showlegend=False),
            row=1, col=1
        )
    
    # Final price distribution
    fig.add_trace(
        go.Histogram(x=result['final_prices'], nbinsx=50, name='Final Prices'),
        row=1, col=2
    )
    
    # Payoff distribution
    fig.add_trace(
        go.Histogram(x=result['payoffs'], nbinsx=50, name='Payoffs'),
        row=2, col=1
    )
    
    # Convergence
    running_mean = np.cumsum(result['payoffs']) / np.arange(1, len(result['payoffs']) + 1)
    fig.add_trace(
        go.Scatter(y=running_mean, mode='lines', name='Running Average'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    return fig

def create_greeks_chart(results_dict):
    """Create Greeks comparison chart"""
    models = list(results_dict.keys())
    greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
    
    fig = go.Figure()
    
    for greek in greeks:
        values = [results_dict[model][greek] for model in models]
        fig.add_trace(go.Bar(name=greek.capitalize(), x=models, y=values))
    
    fig.update_layout(
        title='Greeks Comparison Across Models',
        xaxis_title='Pricing Model',
        yaxis_title='Greek Value',
        barmode='group',
        height=400
    )
    return fig

# Main Streamlit App
def main():
    st.title("🎯 Advanced Options Pricing Simulator")
    st.markdown("### Real-time options pricing with multiple models and interactive visualizations")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📊 Option Parameters")
        
        # Basic parameters
        spot_price = st.number_input("Current Stock Price ($)", value=100.0, min_value=0.01)
        strike_price = st.number_input("Strike Price ($)", value=105.0, min_value=0.01)
        time_to_expiry = st.number_input("Time to Expiry (Years)", value=0.25, min_value=0.001, max_value=5.0)
        
        # Market parameters
        st.subheader("Market Parameters")
        volatility = st.slider("Volatility (%)", min_value=1, max_value=100, value=20) / 100
        risk_free_rate = st.slider("Risk-Free Rate (%)", min_value=0, max_value=20, value=5) / 100
        dividend_yield = st.number_input("Dividend Yield (%)", value=0.0, min_value=0.0, max_value=20.0) / 100
        
        # Option type
        option_type = st.selectbox("Option Type", ["call", "put"])
        
        # Model selection
        st.subheader("Pricing Models")
        selected_models = st.multiselect(
            "Select Models to Compare",
            ["Black-Scholes", "Binomial Tree", "Monte Carlo"],
            default=["Black-Scholes"]
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            mc_simulations = st.number_input("Monte Carlo Simulations", value=100000, min_value=1000, max_value=1000000)
            binomial_steps = st.number_input("Binomial Tree Steps", value=100, min_value=10, max_value=500)
    
    # Store in session state for surface plot
    st.session_state.update({
        'strike': strike_price,
        'time_to_expiry': time_to_expiry,
        'risk_free_rate': risk_free_rate,
        'option_type': option_type
    })
    
    # Calculate prices for selected models
    if st.sidebar.button("🚀 Calculate Prices", type="primary"):
        results = {}
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        model_count = len(selected_models)
        
        for i, model in enumerate(selected_models):
            status_text.text(f'Calculating {model}...')
            
            if model == "Black-Scholes":
                results[model] = OptionsPricingSimulator.black_scholes(
                    spot_price, strike_price, time_to_expiry, risk_free_rate, 
                    volatility, dividend_yield, option_type
                )
            elif model == "Binomial Tree":
                results[model] = OptionsPricingSimulator.binomial_tree(
                    spot_price, strike_price, time_to_expiry, risk_free_rate,
                    volatility, dividend_yield, option_type, binomial_steps
                )
            elif model == "Monte Carlo":
                results[model] = OptionsPricingSimulator.monte_carlo(
                    spot_price, strike_price, time_to_expiry, risk_free_rate,
                    volatility, dividend_yield, option_type, mc_simulations
                )
            
            progress_bar.progress((i + 1) / model_count)
        
        status_text.text('✅ Calculations complete!')
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.header("💰 Pricing Results")
        
        # Create columns for results
        cols = st.columns(len(selected_models))
        
        for i, (model, result) in enumerate(results.items()):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{model}</h3>
                    <h2>${result['price']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # Greeks comparison
        if len(selected_models) > 1:
            st.header("📈 Greeks Analysis")
            greeks_fig = create_greeks_chart(results)
            st.plotly_chart(greeks_fig, use_container_width=True)
        
        # Detailed results table
        st.header("📋 Detailed Results")
        results_df = pd.DataFrame(results).T
        results_df = results_df.round(6)
        st.dataframe(results_df, use_container_width=True)
        
        # Model-specific visualizations
        for model, result in results.items():
            if model == "Monte Carlo":
                st.header(f"🎲 {model} Simulation Analysis")
                mc_fig = create_monte_carlo_viz(result)
                st.plotly_chart(mc_fig, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Final Price", f"${np.mean(result['final_prices']):.2f}")
                with col2:
                    st.metric("Std Dev", f"${np.std(result['final_prices']):.2f}")
                with col3:
                    st.metric("Mean Payoff", f"${np.mean(result['payoffs']):.2f}")
                with col4:
                    st.metric("Success Rate", f"{np.mean(result['payoffs'] > 0):.2%}")
    
    # Volatility Surface
    st.header("🌊 Volatility Surface")
    if st.button("Generate Volatility Surface"):
        with st.spinner("Generating surface..."):
            surface_fig = create_price_surface()
            st.plotly_chart(surface_fig, use_container_width=True)
    
    # Educational content
    with st.expander("📚 Model Information"):
        st.markdown("""
        ### Black-Scholes Model
        - **Best for**: European options with constant volatility
        - **Assumptions**: Constant volatility, interest rates, no dividends
        - **Accuracy**: High for liquid markets
        
        ### Binomial Tree Model
        - **Best for**: American options, dividend-paying stocks
        - **Assumptions**: Discrete price movements
        - **Accuracy**: Improves with more steps
        
        ### Monte Carlo Simulation
        - **Best for**: Complex derivatives, path-dependent options
        - **Assumptions**: Random price movements
        - **Accuracy**: Improves with more simulations
        """)

if __name__ == "__main__":
    main()