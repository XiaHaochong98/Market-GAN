import os

import pandas as pd
import yfinance as yf

# List of DJIA companies' ticker symbols as of September 2021
djia_tickers = ['MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO',
                'KO', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM',
                'MCD', 'MRK', 'MSFT', 'NKE', 'PG', 'CRM', 'TRV', 'UNH', 'VZ',
                'V', 'WBA', 'WMT', 'DIS']

# Initialize an empty DataFrame to store all the data
all_data = pd.DataFrame()

for ticker in djia_tickers:
    print(f"Downloading {ticker} data...")
    # Download historical data as a DataFrame
    data = yf.download(ticker, start="2000-01-01", end="2023-06-06")
    print(f'{ticker},{data.shape[0]}')
    # Add a column for the ticker
    data['tic'] = ticker
    # Append the data to the main DataFrame
    all_data = all_data.append(data)

# create a folder to store the data
if not os.path.exists('DJI'):
    os.makedirs('DJI')

# Save all data to a CSV file
all_data.to_csv(os.path.join('DJI','DJI_data.csv'))
