import streamlit as st
import pandas as pd
from fantasy_football_optimizer import load_raw_data, engineer_features, optimize_team

# -------------------------------
# Streamlit UI for Fantasy Football Optimizer
# -------------------------------

def main():
    st.set_page_config(page_title="FPL Team Optimizer", layout="wide")
    st.title("⚽ Fantasy Football Team Optimizer")

    # Sidebar controls
    st.sidebar.header("Optimizer Settings")
    budget = st.sidebar.slider("Budget (£m)", min_value=80.0, max_value=120.0, value=100.0, step=0.5)
    st.sidebar.markdown("---")

    # Optimize button
    if st.sidebar.button("🔄 Optimize Team"):
        with st.spinner("Fetching data and computing optimal squad..."):
            # Load and prepare data
            raw = load_raw_data()
            feats = engineer_features(raw['players'], raw['fixtures'])
            # Run optimization
            team = optimize_team(feats, budget=budget)

        # Display results
        st.success("Optimization complete!")
        st.subheader("Selected 15-Player Squad")
        # Map element_type codes to names
        pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        team['position'] = team['element_type'].map(pos_map)
        display_df = team[['first_name', 'second_name', 'position', 'team', 'cost_m', 'total_points']]
        display_df = display_df.rename(columns={
            'first_name': 'First Name',
            'second_name': 'Last Name',
            'team': 'Club ID',
            'cost_m': 'Cost (£m)',
            'total_points': 'Total Points'
        }).reset_index(drop=True)
        st.dataframe(display_df)

        # Download option
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download squad as CSV",
            data=csv,
            file_name='optimized_squad.csv',
            mime='text/csv'
        )
    else:
        st.info("Adjust the budget in the sidebar and click ‘Optimize Team’ to generate your squad.")

if __name__ == '__main__':
    main()
