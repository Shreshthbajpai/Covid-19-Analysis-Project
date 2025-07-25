import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Suppress warnings for cleaner output in a project setting
warnings.filterwarnings('ignore')

print("Starting COVID-19 Data Analysis Project...")

# --- 1. Data Acquisition ---
# URL for the Our World in Data COVID-19 dataset
DATA_URL = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'

try:
    df = pd.read_csv(DATA_URL)
    print(f"Data loaded successfully from {DATA_URL}")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please check your internet connection or the data URL.")
    exit() # Exit if data cannot be loaded

# --- 2. Initial Data Exploration ---
print("\n--- Initial Data Exploration ---")
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Information (data types, non-null counts):")
df.info()

print("\nDescriptive Statistics for numerical columns:")
print(df.describe())

print("\nMissing Values Count (per column):")
print(df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False))

# --- 3. Data Cleaning and Preprocessing ---
print("\n--- Data Cleaning and Preprocessing ---")

# Convert 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'])

# Sort data by location and date for proper time-series analysis
df = df.sort_values(by=['location', 'date']).reset_index(drop=True)

# Filter out aggregate locations (like 'World', 'Europe', 'Asia')
# We'll keep only entries where 'continent' is not null, implying it's a specific country
df_countries = df[df['continent'].notna()].copy()

# Fill missing numerical values for daily metrics with 0 where appropriate
# This assumes that missing daily reports mean zero new cases/deaths for that day.
# For cumulative values, forward fill is better to carry last known value forward.
cols_to_fill_0 = ['new_cases', 'new_deaths', 'new_cases_smoothed', 'new_deaths_smoothed']
for col in cols_to_fill_0:
    if col in df_countries.columns:
        df_countries[col] = df_countries[col].fillna(0)

# For cumulative metrics, forward fill (ffill) and then fill remaining NaNs with 0
# This is crucial because a missing total case value should carry forward the last known total.
cols_to_ffill = ['total_cases', 'total_deaths', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']
for col in cols_to_ffill:
    if col in df_countries.columns:
        df_countries[col] = df_countries.groupby('location')[col].ffill().fillna(0)

# Create Derived Metrics
# Case Fatality Rate (CFR): total_deaths / total_cases * 100
# Handle division by zero and inf values
df_countries['case_fatality_rate'] = (df_countries['total_deaths'] / df_countries['total_cases']) * 100
df_countries['case_fatality_rate'] = df_countries['case_fatality_rate'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Vaccination Rate (people vaccinated per hundred of population)
df_countries['vaccination_rate_per_hundred'] = (df_countries['people_vaccinated'] / df_countries['population']) * 100
df_countries['vaccination_rate_per_hundred'] = df_countries['vaccination_rate_per_hundred'].replace([np.inf, -np.inf], np.nan).fillna(0)

# People fully vaccinated per hundred of population
df_countries['fully_vaccinated_per_hundred'] = (df_countries['people_fully_vaccinated'] / df_countries['population']) * 100
df_countries['fully_vaccinated_per_hundred'] = df_countries['fully_vaccinated_per_hundred'].replace([np.inf, -np.inf], np.nan).fillna(0)

print("Data preprocessing complete.")

# --- 4. Exploratory Data Analysis (EDA) & Visualization ---
print("\n--- Generating Visualizations ---")

# Ensure Plotly is set to render in browser for VS Code environment
# This might vary based on your VS Code setup (e.g., if you're using Jupyter Notebooks within VS Code)
# If plots don't show, you might need to run this in a Jupyter Notebook environment.
# For basic script execution, plotly will often open a browser tab.

# Global Trends
# Filter data for 'World' aggregate if it exists, otherwise aggregate manually
df_world = df[df['location'] == 'World'].copy()
if df_world.empty:
    print("Warning: 'World' aggregate not found. Aggregating data for global trends...")
    # Manually aggregate world data if 'World' entry is missing
    df_world = df_countries.groupby('date').agg(
        total_cases=('total_cases', 'sum'),
        total_deaths=('total_deaths', 'sum'),
        new_cases=('new_cases', 'sum'),
        new_deaths=('new_deaths', 'sum'),
        total_vaccinations=('total_vaccinations', 'sum')
    ).reset_index()
    # Recalculate smoothed values for the aggregated world data
    df_world['new_cases_smoothed_7_day'] = df_world['new_cases'].rolling(window=7, min_periods=1).mean()
    df_world['new_deaths_smoothed_7_day'] = df_world['new_deaths'].rolling(window=7, min_periods=1).mean()
else:
    # Use existing smoothed data if available, otherwise calculate
    if 'new_cases_smoothed' in df_world.columns:
        df_world['new_cases_smoothed_7_day'] = df_world['new_cases_smoothed']
    else:
        df_world['new_cases_smoothed_7_day'] = df_world['new_cases'].rolling(window=7, min_periods=1).mean()

    if 'new_deaths_smoothed' in df_world.columns:
        df_world['new_deaths_smoothed_7_day'] = df_world['new_deaths_smoothed']
    else:
        df_world['new_deaths_smoothed_7_day'] = df_world['new_deaths'].rolling(window=7, min_periods=1).mean()

# Global Daily New Cases (7-Day Rolling Average)
if not df_world.empty:
    fig = px.line(df_world, x='date', y='new_cases_smoothed_7_day',
                  title='Global Daily New Cases (7-Day Rolling Average)',
                  labels={'new_cases_smoothed_7_day': '7-Day Avg New Cases'},
                  template='plotly_white')
    fig.update_layout(xaxis_title='Date', yaxis_title='Number of Cases')
    fig.show()

    # Global Daily New Deaths (7-Day Rolling Average)
    fig = px.line(df_world, x='date', y='new_deaths_smoothed_7_day',
                  title='Global Daily New Deaths (7-Day Rolling Average)',
                  labels={'new_deaths_smoothed_7_day': '7-Day Avg New Deaths'},
                  template='plotly_white')
    fig.update_layout(xaxis_title='Date', yaxis_title='Number of Deaths')
    fig.show()

    # Global Total Vaccinations Over Time
    fig = px.line(df_world, x='date', y='total_vaccinations',
                  title='Global Total Vaccinations Over Time',
                  labels={'total_vaccinations': 'Total Vaccine Doses Administered'},
                  template='plotly_white')
    fig.update_layout(xaxis_title='Date', yaxis_title='Total Doses')
    fig.show()
else:
    print("Cannot generate global trend plots due to missing 'World' data or insufficient aggregated data.")

# --- Top N Countries Analysis (Bar Charts & Line Plots) ---
latest_data = df_countries.loc[df_countries.groupby('location')['date'].idxmax()]

# Top 10 Countries by Total Confirmed Cases
top_cases = latest_data.nlargest(10, 'total_cases')
fig = px.bar(top_cases, x='location', y='total_cases',
             title='Top 10 Countries by Total Confirmed COVID-19 Cases',
             labels={'location': 'Country', 'total_cases': 'Total Cases'},
             template='plotly_white', color='total_cases', color_continuous_scale=px.colors.sequential.Plasma)
fig.show()

# Top 10 Countries by Total Deaths
top_deaths = latest_data.nlargest(10, 'total_deaths')
fig = px.bar(top_deaths, x='location', y='total_deaths',
             title='Top 10 Countries by Total COVID-19 Deaths',
             labels={'location': 'Country', 'total_deaths': 'Total Deaths'},
             template='plotly_white', color='total_deaths', color_continuous_scale=px.colors.sequential.Plasma)
fig.show()

# Top 10 Countries by Fully Vaccinated Population Percentage
top_fully_vaccinated = latest_data.nlargest(10, 'fully_vaccinated_per_hundred')
fig = px.bar(top_fully_vaccinated, x='location', y='fully_vaccinated_per_hundred',
             title='Top 10 Countries by Fully Vaccinated Population (%)',
             labels={'location': 'Country', 'fully_vaccinated_per_hundred': '% Fully Vaccinated'},
             template='plotly_white', color='fully_vaccinated_per_hundred', color_continuous_scale=px.colors.sequential.Viridis)
fig.show()

# Compare cases/deaths for a few selected countries
selected_countries = ['United States', 'India', 'Brazil', 'United Kingdom', 'France', 'Germany']
df_selected = df_countries[df_countries['location'].isin(selected_countries)]

if not df_selected.empty:
    fig = px.line(df_selected, x='date', y='new_cases_smoothed', color='location',
                  title='Daily New Cases (7-Day Smoothed) in Selected Countries',
                  labels={'new_cases_smoothed': '7-Day Avg New Cases', 'location': 'Country'},
                  template='plotly_white')
    fig.show()

    fig = px.line(df_selected, x='date', y='new_deaths_smoothed', color='location',
                  title='Daily New Deaths (7-Day Smoothed) in Selected Countries',
                  labels={'new_deaths_smoothed': '7-Day Avg New Deaths', 'location': 'Country'},
                  template='plotly_white')
    fig.show()
else:
    print("No data found for selected countries for comparison.")

# --- Geographical Distribution (Choropleth Maps) ---
# World map of total cases
if 'iso_code' in latest_data.columns and 'total_cases' in latest_data.columns:
    fig = px.choropleth(latest_data,
                        locations="iso_code",
                        color="total_cases",
                        hover_name="location",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title='World Map of Total Confirmed COVID-19 Cases (Latest Data)',
                        labels={'total_cases': 'Total Cases'},
                        template='plotly_white')
    fig.show()
else:
    print("Cannot generate world map for total cases (missing 'iso_code' or 'total_cases' column).")

# World map of total deaths
if 'iso_code' in latest_data.columns and 'total_deaths' in latest_data.columns:
    fig = px.choropleth(latest_data,
                        locations="iso_code",
                        color="total_deaths",
                        hover_name="location",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title='World Map of Total COVID-19 Deaths (Latest Data)',
                        labels={'total_deaths': 'Total Deaths'},
                        template='plotly_white')
    fig.show()
else:
    print("Cannot generate world map for total deaths (missing 'iso_code' or 'total_deaths' column).")


# World map of vaccination rate (at least one dose)
if 'iso_code' in latest_data.columns and 'vaccination_rate_per_hundred' in latest_data.columns:
    fig = px.choropleth(latest_data,
                        locations="iso_code",
                        color="vaccination_rate_per_hundred",
                        hover_name="location",
                        color_continuous_scale=px.colors.sequential.Viridis,
                        title='World Map of People Vaccinated against COVID-19 (%) (Latest Data)',
                        labels={'vaccination_rate_per_hundred': '% Vaccinated (at least one dose)'},
                        template='plotly_white')
    fig.show()
else:
    print("Cannot generate world map for vaccination rate (missing 'iso_code' or 'vaccination_rate_per_hundred' column).")


# --- Relationship Analysis (Scatter Plots) ---
# Snapshot for latest available data for correlations
snapshot_date = df_countries['date'].max()
snapshot_data = df_countries[df_countries['date'] == snapshot_date].copy()

# Filter for relevant columns and drop NaNs for scatter plots
correlation_data = snapshot_data.dropna(subset=['stringency_index', 'new_cases', 'case_fatality_rate', 'median_age', 'population'])

if not correlation_data.empty:
    # Stringency Index vs. New Cases
    fig = px.scatter(correlation_data, x='stringency_index', y='new_cases',
                     size='population', color='continent', hover_name='location',
                     title=f'Stringency Index vs. New Cases (Snapshot: {snapshot_date.strftime("%Y-%m-%d")})',
                     labels={'stringency_index': 'Government Stringency Index', 'new_cases': 'New Cases'},
                     template='plotly_white', log_y=True) # Log scale for new_cases helps visualize spread
    fig.show()

    # Median Age vs. Case Fatality Rate
    fig = px.scatter(correlation_data, x='median_age', y='case_fatality_rate',
                     size='population', color='continent', hover_name='location',
                     title=f'Median Age vs. Case Fatality Rate (Snapshot: {snapshot_date.strftime("%Y-%m-%d")})',
                     labels={'median_age': 'Median Age', 'case_fatality_rate': 'Case Fatality Rate (%)'},
                     template='plotly_white')
    fig.show()
else:
    print("Not enough data for correlation plots (stringency index, median age, new cases, CFR).")

# --- 5. Summary and Insights ---
print("\n--- Project Summary ---")
print("This analysis provides a visual overview of the COVID-19 pandemic.")
print("\nKey observations (based on the generated graphs):")
print("1. **Global Trends:** Observe the peaks and troughs in daily new cases and deaths globally, and the steady increase in vaccination efforts.")
print("2. **Country Comparison:** Identify countries most affected by total cases and deaths, and compare daily trends in selected regions.")
print("3. **Geographical Impact:** The choropleth maps highlight the uneven distribution of cases, deaths, and vaccination progress across the globe.")
print("4. **Correlations:** Scatter plots offer initial insights into relationships, such as the (expected) link between median age and case fatality rate, or the (less straightforward) relationship between government stringency and new cases.")
print("\nFurther analysis could include:")
print("- Analyzing specific waves or periods of the pandemic.")
print("- Investigating the impact of specific policy interventions.")
print("- Building predictive models for future trends.")
print("- Deeper dives into specific country data and local factors.")

print("\nData analysis complete. All generated plots should be displayed in your browser or plot viewer.")