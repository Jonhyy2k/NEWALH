import requests
import pandas as pd
import time
import numpy as np

class AlphaVantageClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.last_request_time = 0
        
    def get_stock_data(self, symbol, output_size="full"):
        """
        Get daily stock data from Alpha Vantage
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
        output_size: str
            'compact' for last 100 data points, 'full' for all data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with stock data
        """
        # Rate limiting - ensure at least 12 seconds between requests (5 requests per minute)
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < 12:
            time.sleep(12 - time_since_last_request)
        
        # Build request parameters
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": output_size,
            "datatype": "json",
            "apikey": self.api_key
        }
        
        # Make the request
        print(f"[INFO] Fetching data for {symbol} from Alpha Vantage")
        response = requests.get(self.base_url, params=params)
        self.last_request_time = time.time()
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"[ERROR] API request failed: {response.status_code} - {response.text}")
            return None
        
        # Parse the response
        data = response.json()
        
        # Check for error messages
        if "Error Message" in data:
            print(f"[ERROR] API error: {data['Error Message']}")
            return None
        
        # Extract time series data
        if "Time Series (Daily)" not in data:
            print(f"[WARNING] No time series data found in response: {data}")
            return None
            
        time_series = data["Time Series (Daily)"]
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")
        
        # Rename columns to match the expected format used by the existing code
        df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "4. close",  # Keep this as "4. close" to match existing code
            "5. volume": "volume"
        }, inplace=True)
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Sort index in ascending order and set proper datetime index
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        print(f"[INFO] Retrieved {len(df)} days of data for {symbol}")
        
        return df
    
    def get_symbol_search(self, keywords):
        """
        Search for company symbols with Alpha Vantage
        
        Parameters:
        -----------
        keywords: str
            Search keywords
            
        Returns:
        --------
        list
            List of matching symbols
        """
        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < 12:
            time.sleep(12 - time_since_last_request)
        
        # Build request parameters
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keywords,
            "datatype": "json",
            "apikey": self.api_key
        }
        
        # Make the request
        print(f"[INFO] Searching for '{keywords}' on Alpha Vantage")
        response = requests.get(self.base_url, params=params)
        self.last_request_time = time.time()
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"[ERROR] API request failed: {response.status_code} - {response.text}")
            return None
        
        # Parse the response
        data = response.json()
        
        # Check for error messages
        if "Error Message" in data:
            print(f"[ERROR] API error: {data['Error Message']}")
            return None
        
        # Extract matches
        if "bestMatches" not in data or not data["bestMatches"]:
            print(f"[WARNING] No matches found for '{keywords}'")
            return []
            
        # Return list of symbols with descriptions
        results = []
        for match in data["bestMatches"]:
            results.append({
                "symbol": match.get("1. symbol", ""),
                "name": match.get("2. name", ""),
                "type": match.get("3. type", ""),
                "region": match.get("4. region", "")
            })
        
        return results
    
    def get_company_overview(self, symbol):
        """
        Get company overview data from Alpha Vantage
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
            
        Returns:
        --------
        dict
            Company overview data
        """
        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < 12:
            time.sleep(12 - time_since_last_request)
        
        # Build request parameters
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        # Make the request
        print(f"[INFO] Fetching company overview for {symbol}")
        response = requests.get(self.base_url, params=params)
        self.last_request_time = time.time()
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"[ERROR] API request failed: {response.status_code} - {response.text}")
            return None
        
        # Parse the response
        data = response.json()
        
        # Check if data is empty or has error
        if not data or "Error Message" in data:
            print(f"[WARNING] No company overview data found for {symbol}")
            return None
            
        return data
    
    def get_global_quote(self, symbol):
        """
        Get current quote for a symbol
        
        Parameters:
        -----------
        symbol: str
            Stock symbol
            
        Returns:
        --------
        dict
            Quote data
        """
        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < 12:
            time.sleep(12 - time_since_last_request)
        
        # Build request parameters
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        # Make the request
        print(f"[INFO] Fetching current quote for {symbol}")
        response = requests.get(self.base_url, params=params)
        self.last_request_time = time.time()
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"[ERROR] API request failed: {response.status_code} - {response.text}")
            return None
        
        # Parse the response
        data = response.json()
        
        # Check if data is empty or has error
        if not data or "Error Message" in data or "Global Quote" not in data:
            print(f"[WARNING] No quote data found for {symbol}")
            return None
            
        quote_data = data["Global Quote"]
        
        # Create simplified quote dictionary
        quote = {
            "symbol": quote_data.get("01. symbol", ""),
            "price": float(quote_data.get("05. price", 0)),
            "change": float(quote_data.get("09. change", 0)),
            "change_percent": quote_data.get("10. change percent", "")
        }
            
        return quote