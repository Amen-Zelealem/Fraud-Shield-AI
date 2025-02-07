import pandas as pd
import socket
import struct
import logging

class FraudIPAnalyzer:
    def __init__(self, fraud_df, ip_country_csv, logger=None):
        self.fraud_df = fraud_df
        self.ip_country_csv = ip_country_csv
        self.logger = logger or logging.getLogger(__name__)  

    def ip_to_int(self, ip):
        """Convert an IP address to an integer representation."""
        try:
            return struct.unpack("!I", socket.inet_aton(ip))[0]
        except socket.error:
            return None

    def process_data(self):
        """Processes the fraud dataset by merging it with IP country data."""
        if self.logger:
            self.logger.info("Loading IP country data...")

        ip_country_data = pd.read_csv(self.ip_country_csv)

        # Convert IP addresses in fraud data to integer
        self.fraud_df['ip_int'] = self.fraud_df['ip_address'].apply(lambda x: self.ip_to_int(str(int(x))) if not pd.isna(x) else None)

        # Drop invalid IPs
        self.fraud_df.dropna(subset=['ip_int'], inplace=True)

        # Convert bounds in IP country data
        ip_country_data['lower_bound_ip_address'] = ip_country_data['lower_bound_ip_address'].astype(int)
        ip_country_data['upper_bound_ip_address'] = ip_country_data['upper_bound_ip_address'].astype(int)

        # Sort for merge_asof
        self.fraud_df.sort_values('ip_int', inplace=True)
        ip_country_data.sort_values('lower_bound_ip_address', inplace=True)

        # Merge datasets
        merged_data = pd.merge_asof(
            self.fraud_df,
            ip_country_data,
            left_on='ip_int',
            right_on='lower_bound_ip_address',
            direction='backward'
        )

        # Filter valid IP ranges
        merged_data = merged_data[
            (merged_data['ip_int'] >= merged_data['lower_bound_ip_address']) &
            (merged_data['ip_int'] <= merged_data['upper_bound_ip_address'])
        ]

        # Drop unnecessary columns
        merged_data.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True)

        if self.logger:
            self.logger.info("Data processing complete.")

        return merged_data
