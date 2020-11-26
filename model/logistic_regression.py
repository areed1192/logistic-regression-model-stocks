import json
import pprint
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from configparser import ConfigParser
from td.client import TDClient

from indicators import ema
from indicators import macd

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def grab_candle_data(pull_from_td: bool) -> list[dict]:
    """A function that grabs candle data from TD Ameritrade,
    cleans up the data, and saves it to a JSON file, so we can
    use it later.

    ### Parameters
    ----------
    pull_from_td : bool
        If `True`, pull fresh candles from the TD
        Ameritrade API. If `False`, load the data
        from the JSON file.

    ### Returns
    -------
    list[dict]
        A list of candle dictionaries with cleaned
        dates, and additional values.
    """

    if pull_from_td:

        # Grab configuration values.
        config = ConfigParser()
        config.read('config/config.ini')

        # Read the Config File.
        CLIENT_ID = config.get('main', 'CLIENT_ID')
        REDIRECT_URI = config.get('main', 'REDIRECT_URI')
        JSON_PATH = config.get('main', 'JSON_PATH')
        ACCOUNT_NUMBER = config.get('main', 'ACCOUNT_NUMBER')

        # Create a new session
        TDSession = TDClient(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            credentials_path=JSON_PATH,
            account_number=ACCOUNT_NUMBER
        )

        # Login to the session
        TDSession.login()

        # Initialize the list to store candles.
        all_candles = []

        # Loop through each Ticker.
        for ticker in ['AAPL', 'NIO', 'FIT', 'TSLA', 'MSFT', 'AMZN', 'IBM']:

            # Grab the Quotes.
            quotes = TDSession.get_price_history(
                symbol=ticker,
                period_type='day',
                period='10',
                frequency_type='minute',
                frequency=1,
                extended_hours=False
            )

            # Grab the Candles.
            candles = quotes['candles']

            # Loop through each candle.
            for candle in candles:

                # Calculate the Range.
                candle['range'] = round(candle['high'] - candle['low'], 5)

                # Add the Symbol.
                candle['symbol'] = quotes['symbol']

                # Convert to ISO String.
                candle['datetime_iso'] = datetime.fromtimestamp(
                    candle['datetime']/1000
                ).isoformat()

                # Conver to a Timestamp non-milliseconds.
                candle['datetime_non_milli'] = int(candle['datetime'] / 1000)

                all_candles.append((candle))

        # Save it to a JSON File.
        with open(file='data/candles.json', mode='w+') as candle_file:
            json.dump(obj=all_candles, fp=candle_file, indent=4)

    elif pull_from_td is False and pathlib.Path('data/candles.json').exists():

        # Save it to a JSON File.
        with open(file='data/candles.json', mode='r') as candle_file:
            all_candles = json.load(fp=candle_file)

    return all_candles


def create_stock_frame(candles: list) -> pd.DataFrame:
    """Takes our list of candles and converts it to a Pandas
    Dataframe.

    ### Parameters
    ----------
    candles : list
        A list of candle dictionaries.

    ### Returns
    -------
    pd.DataFrame
        A multi-index Pandas Dataframe, that is sorted
        by the ticker symbol and datetime.
    """

    # Create the Dataframe.
    price_df = pd.DataFrame(data=candles)

    # Parse the Datetime column.
    price_df['datetime'] = pd.to_datetime(
        price_df['datetime'],
        unit='ms',
        origin='unix'
    )

    # Set the multi-index.
    price_df = price_df.set_index(
        keys=['symbol', 'datetime']
    )

    # Sort it.
    price_df = price_df.sort_index()

    return price_df


def add_dataframe_indicators(stock_frame: pd.DataFrame) -> pd.DataFrame:
    """Adds the different technical indicator to our DataFrame.

    ### Parameters
    ----------
    stock_frame : pd.DataFrame
        A pandas DataFrame with the Open, Close,
        High, and Low price.

    ### Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with the indicators added
        to the data frame.
    """

    # Add the MACD Indicator.
    stock_frame = macd(
        stock_frame=stock_frame,
        fast_period=12,
        slow_period=26
    )

    # Add the EMA Indicator.
    stock_frame = ema(
        stock_frame=stock_frame,
        period=50,
        alpha=1.0
    )

    return stock_frame


def add_dependent_variable(stock_frame: pd.DataFrame) -> pd.DataFrame:
    """Adds the dependent variable to to the Stock Frame.

    ### Overview:
    ----------
    Create a new column in our dataframe that will be used
    for predictive purposes. Using the `numpy.where` function
    we will create a column where `1` denotes that the close
    of the next data (t + 1) is `HIGHER` than the actual day
    (t + 0).

    ### Parameters
    ----------
    stock_frame : pd.DataFrame
        A pandas DataFrame with the Open, Close,
        High, and Low price.

    ### Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with the dependent variable
        column defined.
    """

    # Add the column.
    stock_frame['Y'] = np.where(
        stock_frame['close'].shift(-1) > stock_frame['close'],
        1, -1
    )

    # Grab the first timestamp for the first Stock, WE ARE
    # ASSUMING EVERYONE HAS THE SAME TIMESTAMPS.
    drop_stamp = stock_frame.index.values[-1][1]

    # Sort it one last time.
    stock_frame.sort_index(
        level=['symbol','datetime'],
        inplace=True
    )

    # Drop the first row from each.
    stock_frame.drop(
        labels=drop_stamp,
        level='datetime',
        inplace=True
    )

    return stock_frame


def split_and_fit(stock_frame: pd.DataFrame, split: float = 0.80) -> tuple:
    """Splits the dataset and fits it to the model.

    ### Parameters
    ----------
    stock_frame : pd.DataFrame
        The dataframe containing the data we
        want to split and fit.

    split : float, optional
        How much to split the sample by, by
        default 0.80

    ### Returns
    -------
    tuple
        A tuple with the different splits and
        the "fitted" model.
    """

    stock_frame.dropna(inplace=True)

    # Prep the Data.
    X = stock_frame[['open', 'high', 'low', 'ema', 'macd', 'range']]
    y = stock_frame['Y']

    # Split the Dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split, random_state=42
    )

    # Initialize a new instance of the Model.
    model = LogisticRegression()

    # Fit the data.
    model = model.fit(X_train, y_train)

    return (X_train, X_test, y_train, y_test, model)


def evaluate_model(model_obj: LogisticRegression, X_test, Y_test) -> None:
    """Returns the Score for the LogisticRegression Model.

    ### Parameters
    ----------
    model_obj : LogisticRegression
        The logistic regression model we are
        evalutating.

    X_test : pd.DataFrame
        The test dataset.

    y_test : pd.Series
        The test predictions.
    """

    # Grab the Score.
    model_score = model_obj.score(
        X=X_test,
        y=Y_test
    )

    print(model_score)


def plot_confusion_matrix(model_obj: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series, show_plot: bool) -> None:
    """Creates a confusion matrix and then plots it.

    ### Parameters
    ----------
    model_obj : LogisticRegression
        The logistic regression model we are
        evalutating.

    X_test : pd.DataFrame
        The test dataset.

    y_test : pd.Series
        The test predictions.
    
    show_plot : bool
        If `True` it will show the created
        plot. If `False` it will not show it.
    """

    # Grab the Score.
    model_score = model_obj.score(
        X=X_test,
        y=y_test
    )

    # Define some predicitions.
    predictions = model_obj.predict(X_test)

    # Define the confusion matrix.
    confusion_matrix = metrics.confusion_matrix(
        y_true=y_test,
        y_pred=predictions
    )

    print("CONFUSION MATRIX")
    print("-"*20)
    print(confusion_matrix)

    # Define the plot.
    plt.figure(figsize=(9, 9))

    # Create a heatmap for the matrix.
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".3f",
        linewidths=.5,
        square=True,
        cmap='Blues_r'
    )

    # Define the Title.
    all_sample_title = 'Accuracy Score: {0}'.format(model_score)

    # Set the Title.
    plt.title(
        label=all_sample_title,
        size=15
    )

    # Add the Axis Labels.
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    if show_plot:
        plt.show()


if __name__ == '__main__':

    # Grab the Candles.
    candles = grab_candle_data(pull_from_td=False)

    # Conver it to a dataframe.
    stock_frame = create_stock_frame(candles=candles)

    # Add the `DataFrame`.
    stock_frame_with_indicators = add_dataframe_indicators(
        stock_frame=stock_frame
    )

    # Add the Dependent Column.
    stock_frame_with_y = add_dependent_variable(
        stock_frame=stock_frame_with_indicators
    )

    # Dump it to a CSV file.
    stock_frame_with_y.to_csv('input_data.csv')

    # Fit the data.
    model_info = split_and_fit(
        stock_frame=stock_frame_with_y,
        split=.7
    )

    # Evaluate the model.
    evaluate_model(
        model_obj=model_info[4],
        X_test=model_info[1],
        Y_test=model_info[3]
    )

    # Plot a confusion matrix.
    plot_confusion_matrix(
        model_obj=model_info[4],
        X_test=model_info[1],
        y_test=model_info[3],
        show_plot=False
    )


