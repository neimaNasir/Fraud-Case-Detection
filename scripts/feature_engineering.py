import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class FeatureEngineering:
    def __init__(self, data: pd.DataFrame, logging):
        """
        Initializes the FeatureEngineering class with the transaction data DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing the transaction data.
        """
        self.data = data.copy()
        self.processed_data = None
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.logging = logging
        self.logging.info("FeatureEngineering class initialized with the provided DataFrame.")

    def preprocess_datetime(self):
        """
        Converts 'signup_time' and 'purchase_time' columns to datetime format and creates time-based features.
        """
        self.logging.info("Preprocessing datetime features...")
        try:
            self.data['signup_time'] = pd.to_datetime(self.data['signup_time'])
            self.data['purchase_time'] = pd.to_datetime(self.data['purchase_time'])

            # Create time-based features
            self.data['hour_of_day'] = self.data['purchase_time'].dt.hour
            self.data['day_of_week'] = self.data['purchase_time'].dt.dayofweek
            self.data['purchase_delay'] = (self.data['purchase_time'] - self.data['signup_time']).dt.total_seconds() / 3600  # Time difference in hours
            self.logging.info("Datetime features successfully created.")
        except Exception as e:
            self.logging.error("Error in preprocessing datetime features: %s", e)
            raise

    def calculate_transaction_frequency(self):
        """
        Calculates the transaction frequency and velocity for each user and device.
        """
        self.logging.info("Calculating transaction frequency and velocity...")
        try:
            # Transaction frequency per user
            user_freq = self.data.groupby('user_id').size()
            self.data['user_transaction_frequency'] = self.data['user_id'].map(user_freq)

            # Transaction frequency per device
            device_freq = self.data.groupby('device_id').size()
            self.data['device_transaction_frequency'] = self.data['device_id'].map(device_freq)

            # Transaction velocity: transactions per hour for each user
            self.data['user_transaction_velocity'] = self.data['user_transaction_frequency'] / self.data['purchase_delay']
            self.logging.info("Transaction frequency and velocity calculated successfully.")
        except Exception as e:
            self.logging.error("Error in calculating transaction frequency and velocity: %s", e)
            raise

    def normalize_and_scale(self):
        """
        Normalizes and scales numerical features using StandardScaler.
        Applies scaling to selected columns and stores the transformed DataFrame.
        """
        self.logging.info("Normalizing and scaling numerical features...")
        try:
            numerical_features = ['purchase_value', 'user_transaction_frequency', 'device_transaction_frequency', 
                                'user_transaction_velocity', 'hour_of_day', 'day_of_week', 'purchase_delay', 'age']
            self.data[numerical_features] = self.scaler.fit_transform(self.data[numerical_features])
            self.logging.info("Numerical features normalized and scaled successfully.")
        except Exception as e:
            self.logging.error("Error in normalizing and scaling numerical features: %s", e)
            raise

    def encode_categorical_features(self):
        """
        Encodes categorical features such as 'source', 'browser', and 'sex' using one-hot encoding.
        """
        self.logging.info("Encoding categorical features...")
        try:
            categorical_features = ['source', 'browser', 'sex']
            self.data = pd.get_dummies(self.data, columns=categorical_features, drop_first=True)
            self.logging.info("Categorical features encoded successfully.")
        except Exception as e:
            self.logging.error("Error in encoding categorical features: %s", e)
            raise

    def pipeline(self):
        """
        Executes the full feature engineering pipeline, including time-based feature extraction, 
        transaction frequency/velocity calculation, normalization, scaling, and encoding categorical features.
        """
        self.logging.info("Starting the feature engineering pipeline...")
        try:
            self.preprocess_datetime()
            self.calculate_transaction_frequency()
            self.normalize_and_scale()
            self.encode_categorical_features()
            self.processed_data = self.data
            self.logging.info("Feature engineering pipeline executed successfully.")
        except Exception as e:
            self.logging.error("Error in the feature engineering pipeline: %s", e)
            raise

    def get_processed_data(self) -> pd.DataFrame:
        """
        Returns the processed DataFrame with all the engineered features.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        self.logging.info("Retrieving processed data...")
        if self.processed_data is None:
            self.logging.error("Data has not been processed. Run the pipeline() method first.")
            raise ValueError("Data has not been processed. Run the pipeline() method first.")
        self.logging.info("Processed data retrieved successfully.")
        return self.processed_data