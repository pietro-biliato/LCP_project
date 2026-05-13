import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import yfinance as yf
import mplfinance as mpf
from hurst import compute_Hc
from scipy.fftpack import fft, ifft, fftfreq
from scipy.stats import norm
from pymannkendall import original_test, yue_wang_modification_test, yue_wang_modification_test
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
from scipy.stats import anderson
from scipy.special import erf
import matplotlib.lines as mlines



def import_data(file_path):
    """
    function to import the dataset
    """
    df = pd.read_csv(file_path, sep="\t")

    # Rename columns
    df.rename(columns={
        '<DATE>': 'Date', '<TIME>': 'Time', '<OPEN>': 'Open', '<HIGH>': 'High',
        '<LOW>': 'Low', '<CLOSE>': 'Close', '<TICKVOL>': 'TickVol', '<VOL>': 'Volume', '<SPREAD>': 'Spread'
    }, inplace=True)

    # Convert to datetime index
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
    df.set_index('datetime', inplace=True)
    df = df[df.index.dayofweek < 5]

    # Heikin-Ashi Calculation
    df['HA_close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['HA_open']  = df['Open']
    df['HA_high']  = df[['High', 'HA_open', 'HA_close']].max(axis=1)
    df['HA_low']   = df[['Low', 'HA_open', 'HA_close']].min(axis=1)

    return df, df['HA_close'] #Returns both the entire dataset and just close


def interval_selector(n_s, n_e, df):
    """
    function to select the interval of data to consider
    """
    starting_position=96*n_s  #2350
    print(df.index[starting_position])
    if (starting_position!=0):
        start = df.index[starting_position]
    else:
        start=pd.to_datetime(df.index[0])
    start_index=df.index.get_loc(start)
    print(df.index.get_loc(start))
    end_index=96*n_e
    if (end_index!=0):
        end = df.index[end_index]
        print(end)
    else:
        end = pd.to_datetime('2020.10.12 08:45:00')
        end_index=df.index.get_loc(end)
    return start_index, end_index


def plot_close(data, start=None, end=None):
    """
    Function to plot the closing prices over a specified time period.

    Parameters:
    - data: pandas Series, the time series of closing prices
    - start: str, datetime, or int, start of the time period (optional)
    - end: str, datetime, or int, end of the time period (optional)
    """
    # Filter data based on the type of start and end
    if start is not None or end is not None:
        if isinstance(start, (str, pd.Timestamp)) or isinstance(end, (str, pd.Timestamp)):
            # Use .loc for datetime filtering
            data = data.loc[start:end]
        elif isinstance(start, int) or isinstance(end, int):
            # Use positional slicing
            data = data[start:end]

    # Plot the closing prices
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["<CLOSE>"], label='Closing Price', color='blue')

    # Customize the plot
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Closing Price', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


def filter_signal(sig, datetime_index, time_step=15*60, freq_factor=100,plot=False):
    """
    Filters the input signal by removing high-frequency components based on the peak frequency.

    Parameters:
    sig (array-like): The signal to be filtered.
    datetime_index (pd.DatetimeIndex): The datetime index for the signal.
    time_step (float): The time step (in seconds) between each data point (default is 15 minutes).
    freq_factor (float): The factor used to determine the cutoff frequency for filtering (default is 10).

    Returns:
    filtered_sig_series (pd.Series): A Pandas Series containing the filtered signal with the datetime index.
    """
    # The corresponding frequencies
    sample_freq = fftfreq(sig.size, d=time_step)

    # Perform the FFT
    sig_fft = fft(np.array(sig))
    power = np.abs(sig_fft)
    
    if(plot==True):
        # Plot the power spectrum
        plt.figure(figsize=(6, 5))
        plt.plot(sample_freq, np.log(power))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power')
        plt.title("Power spectrum of the signal")

    # Find the peak frequency (only considering positive frequencies)
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]

    # Filter high frequencies based on the peak frequency
    high_freq_fft = sig_fft.copy()
    high_freq_fft[np.abs(sample_freq) > freq_factor * peak_freq] = 0  # Apply frequency filtering
    filtered_sig = ifft(high_freq_fft)

    # Convert filtered signal to real values (the result of IFFT might be complex)
    filtered_sig = np.real(filtered_sig)

    # Create a Pandas Series with the datetime index and the filtered signal
    filtered_sig_series = pd.Series(filtered_sig, index=datetime_index)

    return filtered_sig_series


def filter_signal_by_auc(sig, datetime_index, time_step=15*60, discard_fraction=0.03,plot_spectrum=False):
    """
    Filters the input signal by removing high-frequency components based on the area under the curve (AUC).

    Parameters:
    sig (array-like): The signal to be filtered.
    datetime_index (pd.DatetimeIndex): The datetime index for the signal.
    time_step (float): The time step (in seconds) between each data point (default is 15 minutes).
    discard_fraction (float): Fraction of the total power to discard (default is 10%).

    Returns:
    filtered_sig_series (pd.Series): A Pandas Series containing the filtered signal with the datetime index.
    """
    # The corresponding frequencies
    sample_freq = fftfreq(sig.size, d=time_step)
    pos_mask = sample_freq > 0
    freqs = sample_freq[pos_mask]

    # Perform the FFT
    sig_fft = fft(np.array(sig))
    power = np.abs(sig_fft) ** 2

    # Compute cumulative power
    cumulative_power = np.cumsum(power[pos_mask])
    total_power = cumulative_power[-1]
    target_power = (1 - discard_fraction) * total_power

    # Determine the cutoff frequency
    cutoff_idx = np.searchsorted(cumulative_power, target_power)
    cutoff_freq = freqs[cutoff_idx]

    # Apply the frequency filter
    filtered_fft = sig_fft.copy()
    filtered_fft[np.abs(sample_freq) > cutoff_freq] = 0
    filtered_sig = ifft(filtered_fft)

    # Convert filtered signal to real values (IFFT output might be complex)
    filtered_sig = np.real(filtered_sig)

    # Create a Pandas Series with the datetime index and the filtered signal
    filtered_sig_series = pd.Series(filtered_sig, index=datetime_index)

    if(plot_spectrum==True):
        # Plot for visualization
        plt.figure(figsize=(6, 5))
        plt.plot(freqs, np.cumsum(power[pos_mask]) / total_power, label="Cumulative Power")
        plt.axvline(cutoff_freq, color='red', linestyle='--', label=f"Cutoff: {cutoff_freq:.2e} Hz")
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Normalized Cumulative Power')
        plt.title("Cumulative Power Spectrum")
        plt.legend()
        plt.show()

    return filtered_sig_series


def get_period(sig,discard_fraction=0.1,time_step=15*60):
    """
    Determines the cutoff frequency for filtering based on the area under the curve (AUC).

    Parameters:
    sig (array-like): The input signal.
    time_step (float): The time step (in seconds) between each data point (default is 15 minutes).
    discard_fraction (float): Fraction of the total power to discard (default is 10%).

    Returns:
    float: The cutoff frequency in Hz.
    """
    # Compute the corresponding frequencies
    sample_freq = fftfreq(len(sig), d=time_step)
    pos_mask = sample_freq > 0
    freqs = sample_freq[pos_mask]

    # Perform FFT
    sig_fft = fft(np.array(sig))
    power = np.abs(sig_fft) ** 2

    # Compute cumulative power
    cumulative_power = np.cumsum(power[pos_mask])
    total_power = cumulative_power[-1]
    target_power = (1 - discard_fraction) * total_power

    # Determine the cutoff frequency
    cutoff_idx = np.searchsorted(cumulative_power, target_power)

    # Return the corresponding frequency
    cutoff_freq= freqs[cutoff_idx] if cutoff_idx < len(freqs) else freqs[-1]

    return int(1/cutoff_freq * 1/time_step)


def mann_kendall(chunk_size, start_index, end_index, df, plotting=False):
    """
    Function to apply the Mann-Kendall test to a given window.
    """
    or_test = []
    modified_test = []

    # Ensure df_filtered is properly assigned
    df_filtered = filter_signal_by_auc(
        df['HA_close'][start_index:end_index].values,
        df[start_index:end_index].index,
        plot_spectrum=False
    )

    for i in range(0, int((end_index - start_index) / chunk_size)):
        s = start_index + i * chunk_size
        e = s + chunk_size

        trend_test_filtered = original_test(df['HA_close'][s:e].values)
        trend_test_modified_filtered = yue_wang_modification_test(df_filtered.iloc[s-start_index:e-start_index].values)

        # Convert test results to numerical values
        or_test.append(1 if trend_test_filtered[0] == 'increasing' else (-1 if trend_test_filtered[0] == 'decreasing' else 0))
        modified_test.append(1 if trend_test_modified_filtered[0] == 'increasing' else (-1 if trend_test_modified_filtered[0] == 'decreasing' else 0))

    if plotting:
        plt.figure(figsize=(14, 5))
        for i in range(len(or_test)):
            s = start_index + i * chunk_size
            e = s + chunk_size

            if e >= len(df_filtered):  # Ensure we don't exceed bounds
                break

            # Determine colors based on test results
            color_map = {1: 'green', -1: 'red', 0: 'blue'}
            color_filtered_map = {1: 'orange', -1: 'black', 0: 'cyan'}

            color = color_map[or_test[i]]
            color_filtered = color_filtered_map[modified_test[i]]

            # Plot original data chunk
            plt.plot(df.index[s:e], df['HA_close'][s:e], color=color, label='_nolegend_')

            # Plot filtered data chunk
            plt.plot(df_filtered.iloc[s-start_index:e-start_index].index,
                     df_filtered.iloc[s-start_index:e-start_index].values,
                     color=color_filtered, linewidth=1, label='_nolegend_')

            # Add vertical line to separate chunks
            plt.axvline(df.index[s], color='grey', linestyle='dashed')
            plt.xticks(rotation=45)

        # Define legend elements
        legend_elements = [
            mlines.Line2D([], [], color='red', linestyle='-', label='Original Test (Decreasing)'),
            mlines.Line2D([], [], color='green', linestyle='-', label='Original Test (Increasing)'),
            mlines.Line2D([], [], color='blue', linestyle='-', label='Original Test (No Trend)'),
            mlines.Line2D([], [], color='black', linestyle='-', label='Modified Test (Decreasing)'),
            mlines.Line2D([], [], color='orange', linestyle='-', label='Modified Test (Increasing)'),
            mlines.Line2D([], [], color='cyan', linestyle='-', label='Modified Test (No Trend)'),
            mlines.Line2D([], [], color='gray', linestyle='--', label='Chunk Separator')
        ]
        
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.title('MK Test Applied to Filtered Data')
        plt.legend(handles=legend_elements, loc='lower right')

    return or_test, modified_test

def heatmap(stride_values, chunk_size, start_index, end_index, df):
    """
    function to partially assess the stability of the test.
    This is done via applying the test on a rolling basis, but starting from a stride between intervals much bigger than 1,
    and then progressively decreasing it.
    """
    test_results_matrix = []
    prices_filtered = filter_signal_by_auc(df['HA_close'][start_index:end_index].values, df[start_index:end_index].index,plot_spectrum=False)
    for rolling_stride in stride_values:
        test_results = []
        z_stat=[]
        significance=[]
        
        for start_idx in range(start_index, end_index - chunk_size + 1, rolling_stride):
            end_idx = start_idx + chunk_size

            # Apply Mann-Kendall Trend Test
            trend_test_modified = yue_wang_modification_test(prices_filtered[start_idx:end_idx])
            significance.append(trend_test_modified[1])
            z_stat.append(trend_test_modified[3])

            # Save test results as numerical values for plotting
            if trend_test_modified[0] == 'no trend':
                test_results.append(0)
            elif trend_test_modified[0] == 'increasing':
                test_results.append(1)
            else:
                test_results.append(-1)

        # Pad test_results to match the maximum length, needed bc we'll convert test_results_matrix into an array --> uniform len needed
        max_length = max(len(tr) for tr in test_results_matrix) if test_results_matrix else len(test_results)
        while len(test_results) < max_length:
            test_results.append(np.nan)  # Use np.nan to indicate missing values

        test_results_matrix.append(test_results)

    test_results_matrix = np.array(test_results_matrix)

    # Plot the heatmap with custom settings
    plt.figure(figsize=(15, 6))
    custom_cmap = ListedColormap(['red', 'blue', 'green'])
    mesh = plt.pcolormesh(
        test_results_matrix,
        cmap=custom_cmap,
        linewidths=0.5,
        edgecolors='black',

    )
    # Add a custom legend
    legend_patches = [
        mpatches.Patch(color='red', label='Decreasing Trend (-1)'),
        mpatches.Patch(color='blue', label='No Trend (0)'),
        mpatches.Patch(color='green', label='Increasing Trend (1)')
    ]
    plt.legend(handles=legend_patches, loc='upper right', title="Trend Type")
    plt.yticks(ticks=np.arange(0.5, len(stride_values)), labels=stride_values)
    plt.xlabel('# of Time Windows', fontsize=14)
    plt.ylabel('Stride Length',fontsize=14)
    plt.title('Trend Detection Heatmap vs. Stride Length',fontsize=16)
    plt.show()

def z_statistic(chunk_size, start_index, end_index, df, gif=False):
    '''
    function to apply the Mann Kendall test on a rolling basis and to properly display the result
    '''
    def compute_stat(x):
        return yue_wang_modification_test(x)[3]
    def compute_significance(x):
        return yue_wang_modification_test(x)[1]
    def p_value_to_z(p_value): #convert the pvalue to the corrisponding value of the statistic
        return norm.ppf(1 - p_value / 2)


    df_filtered = df['HA_close'][start_index:end_index].copy()
    df_filtered = filter_signal_by_auc(df_filtered.values, df_filtered.index)
    df_filtered = df_filtered[start_index+10:end_index-10]
    df_filtered=df_filtered.to_frame(name='HA_close')

    df_filtered['stat']=df_filtered['HA_close'].rolling(window=chunk_size, min_periods=3).apply(compute_stat)
    df_filtered['significance']=df_filtered['HA_close'].rolling(window=chunk_size, min_periods=3).apply(compute_significance)

    significant=df_filtered['stat'][df_filtered['significance'] > 0]
    not_significant=df_filtered['stat'][df_filtered['significance'] < 1]
    z_bound=p_value_to_z(0.05) #0.05 because we choose a CI of 95%

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3,1]})
    fig.tight_layout()

    ax[0].plot(df_filtered['HA_close'])
    xcoords=range(start_index,end_index,chunk_size)
    xcoords=df.index[xcoords]
    for xc in xcoords:
        for a in ax:
            a.axvline(x=xc, color='grey', linestyle='--', linewidth=1, alpha=.5)
            a.grid(visible=False)
    ax[0].set_ylabel('price')

    ax[1].scatter(df_filtered['stat'].index[:],df_filtered['stat'], s=.5, label='z-statistic')
    ax[1].axhline(y=z_bound, color='black', linestyle='--', alpha=.5)
    ax[1].axhline(y=-z_bound, color='black', linestyle='--', alpha=.5, label='CI=95%')
    ax[1].axhspan(z_bound, ax[1].get_ylim()[1], facecolor='green', alpha=0.2)
    ax[1].axhspan(-z_bound, z_bound, facecolor='blue', alpha=0.2)
    ax[1].axhspan(ax[1].get_ylim()[0], -z_bound, facecolor='red', alpha=0.2)
    ax[1].set_ylabel('z-statistic')
    ax[1].legend()

    if(gif==True):
        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3,1]})
        fig.tight_layout()

        def update(frame):
            ax[0].cla()  # Clear previous plot
            ax[1].cla()  # Clear previous plot

            ax[0].set_ylabel('price')
            ax[1].set_ylabel('z-statistic')

            # Highlight rolling window in ax[0] (gray shading and vertical lines)
            start_idx = frame
            end_idx = start_idx + chunk_size
            if end_idx < len(df_filtered):
                ax[0].plot(df_filtered['HA_close'], label="HA_close")
                ax[0].axvspan(df_filtered.index[start_idx], df_filtered.index[end_idx], color='gray', alpha=0.3)
                ax[0].axvline(df_filtered.index[start_idx], color='red', linestyle='--', linewidth=1)
                ax[0].axvline(df_filtered.index[end_idx], color='red', linestyle='--', linewidth=1)

            # In ax[1], plot the computed stats and show window as well
            ax[1].scatter(df_filtered.index[:end_idx], df_filtered['stat'][:end_idx], s=0.5, label="z-statistic")
            if end_idx < len(df_filtered):
                ax[1].axvspan(df_filtered.index[start_idx], df_filtered.index[end_idx], color='gray', alpha=0.3)
            # Show the z-boundaries for significance in ax[1]
            ax[1].axhline(y=z_bound, color='black', linestyle='--', alpha=0.5)
            ax[1].axhline(y=-z_bound, color='black', linestyle='--', alpha=0.5, label='no trend (CI=95%)')
            ax[1].axhspan(z_bound, ax[1].get_ylim()[1], facecolor='green', alpha=0.2)
            ax[1].axhspan(-z_bound, z_bound, facecolor='blue', alpha=0.2)
            ax[1].axhspan(ax[1].get_ylim()[0], -z_bound, facecolor='red', alpha=0.2)
            ax[1].legend()

            return ax

        # Set the number of frames based on the rolling window size and length of df_filtered['stat']
        num_frames = len(df_filtered['stat']) - chunk_size

        # Animation setup
        ani = FuncAnimation(fig, update, frames=range(0, num_frames), interval=200, repeat=False)

        fig.suptitle('Prices vs z-stat')
        ax[0].set_ylabel('price')
        ax[1].set_ylabel('z-statistic')

        # Save the animation as a gif
        ani.save('rolling_computation_evolution.gif', writer='Pillow', fps=9)



###### some of group1 functions:
def assess_normality_rolling(dataset, lambda_ : int, ax = None) -> float:
    x = np.array([0])
    n = lambda_
    # repeat the process n times
    for i in range(n):
        # we will sample the dataset choosing a value every lambda points on average (poisson distr)
        choice = [np.random.randint(lambda_)]  #choice is the vector of indices of chosen values
        for i in range(1, int(len(dataset)/lambda_) ):
            l = np.random.poisson(lambda_)
            if (int(choice[-1]+l) > len(dataset)-1):
                break
            choice.append(int(choice[-1]+l))
        #filtered data, chosen approx every lambda points
        chosen_data = (dataset)[choice]
        x = np.concatenate((x,chosen_data), axis = 0)
    if (ax is not None):
        bin_edges = np.arange(np.min(x), np.max(x), 50)
        ax.hist(x, bins = "auto", density=True)
        ax.set_title(f"Input dataset has {len(dataset)} points"+ '\n'+f"Sampling every ~ {lambda_} points, repeating {n} times " + '\n' + f"Total number of points in the distr: {len(x)}" )

    dev_from_normality = anderson(x).statistic

    # we want to normalize tha value so that it is between 0 and 1
    # when dev is high (high deviations), the process is not very gaussian distributed, so we want to return a value close to 1
    # when dev is low, the process is white noise-like, so we want to return a value close to 0
    #normalized_norm = 2/(1+ np.exp(-dev_from_normality / (0.4*lambda_ ))) - 1    # but maybe work on the normalization
    normalized_norm = erf(dev_from_normality /( 0.4*lambda_))
    return normalized_norm

def find_amplitudes(x : np.array) -> float:
    """
    Find the amplitudes of the peaks in the given time series data.

    Parameters:
     - t (np.array): Array of time values.
     - x (np.array): Array of corresponding data values.
     - factor (float, optional): Factor to adjust the tollerance distance between peaks. Default is 0.3.

    Returns:
     - np.array: Array of amplitudes of the detected peaks.
    """
    from scipy.signal import find_peaks
    factor = 0.3
    dist = factor*np.mean(find_period(x))
    max_peaks, _ = find_peaks(x, distance=dist)
    min_peaks, _ = find_peaks(-x, distance=dist)
    return np.mean(np.concatenate((x[max_peaks], -x[min_peaks])))

def volatility(x : pd.Series) -> float:
    '''Calculate the volatility of a time series usign AR(1) model
     Args:
         x : np.array time series with rectified price
     Returns:
         float : volatility
     '''
    from statsmodels.tsa.arima.model import ARIMA
    AR_model = ARIMA(x, order=(1,0,0), trend='n', enforce_stationarity=False)
    res = AR_model.fit(method='burg')
    AR_phi = res.arparams[0]
    AR_sigma = res.params[-1]

    dt = 1
    theta_AR = -np.log(AR_phi)/dt
    sigma_AR = np.sqrt(AR_sigma * (2 * theta_AR / (1 - AR_phi**2)))

    return sigma_AR

def mean_revertion_index(x : np.array) -> float:
    '''Calculate the mean-reverting index of a time series
    Args:
        x : np.array time series
    Returns:
        float : mean-reverting index from 0 to 1
    '''
    return 1 / ( 1 + np.exp(12*volatility(x) / find_amplitudes(x)-5))

def rectifiy_price(df, column="Close", window=180, polyorder=3):
    from scipy.signal import savgol_filter
    return (df[column] - savgol_filter(df[column], 180, 3)) / df[column]

#########

def process_time_series(
    file_path,
    base_chunk_size,
    start_index,
    end_index,
    time_step=15*60,
    discard_fraction=0.01,
    plot_spectrum=False):


    df = pd.read_csv(file_path, delimiter='\t')

    # Rename columns
    df.rename(columns={
        '<DATE>': 'Date', '<TIME>': 'Time', '<OPEN>': 'Open', '<HIGH>': 'High',
        '<LOW>': 'Low', '<CLOSE>': 'Close', '<TICKVOL>': 'TickVol', '<VOL>': 'Volume', '<SPREAD>': 'Spread'
    }, inplace=True)

    # Convert to datetime index
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
    df.set_index('datetime', inplace=True)
    df = df[df.index.dayofweek < 5]

    # Heikin-Ashi Calculation
    df['HA_close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['HA_open']  = df['Open']
    df['HA_high']  = df[['High', 'HA_open', 'HA_close']].max(axis=1)
    df['HA_low']   = df[['Low', 'HA_open', 'HA_close']].min(axis=1)

    
    # Subset the DataFrame for the given range
    df_range = df.iloc[start_index:end_index].copy()
    
    def filter_signal_by_auc(sig, datetime_index, time_step=15*60, discard_fraction=0.03,plot_spectrum=False):
        # The corresponding frequencies
        sample_freq = fftfreq(sig.size, d=time_step)
        pos_mask = sample_freq > 0
        freqs = sample_freq[pos_mask]

        # Perform the FFT
        sig_fft = fft(np.array(sig))
        power = np.abs(sig_fft) ** 2

        # Compute cumulative power
        cumulative_power = np.cumsum(power[pos_mask])
        total_power = cumulative_power[-1]
        target_power = (1 - discard_fraction) * total_power

        # Determine the cutoff frequency
        cutoff_idx = np.searchsorted(cumulative_power, target_power)
        cutoff_freq = freqs[cutoff_idx]

        # Apply the frequency filter
        filtered_fft = sig_fft.copy()
        filtered_fft[np.abs(sample_freq) > cutoff_freq] = 0
        filtered_sig = ifft(filtered_fft)

        # Convert filtered signal to real values (IFFT output might be complex)
        filtered_sig = np.real(filtered_sig)

        # Create a Pandas Series with the datetime index and the filtered signal
        filtered_sig_series = pd.Series(filtered_sig, index=datetime_index)

        if(plot_spectrum==True):
            # Plot for visualization
            plt.figure(figsize=(6, 5))
            plt.plot(freqs, np.cumsum(power[pos_mask]) / total_power, label="Cumulative Power")
            plt.axvline(cutoff_freq, color='red', linestyle='--', label=f"Cutoff: {cutoff_freq:.2e} Hz")
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Normalized Cumulative Power')
            plt.title("Cumulative Power Spectrum")
            plt.legend()
            plt.show()

        return filtered_sig_series

    def compute_trend_strength(series, window):
        """Computes a rolling trend strength based on the slope of linear regression."""
        slopes = []
        arr = series.values
        for i in range(len(arr) - window + 1):
            x = np.arange(window)
            y = arr[i:i + window]
            slope, _, _, _, _ = linregress(x, y)
            slopes.append(abs(slope))  # Absolute slope as trend strength
        return np.concatenate([[0] * (window - 1), slopes])
    def adaptive_chunk_size(series, base_chunk, max_chunk, trend_threshold):
        """Adjusts chunk size dynamically based on trend strength."""
        trend_strength = compute_trend_strength(series, window=base_chunk)
        chunk_sizes = []
        for t in trend_strength:
            if t > trend_threshold:
                chunk_sizes.append(min(max_chunk, int(base_chunk * (1 + t * 3))))
            else:
                chunk_sizes.append(base_chunk)
        return chunk_sizes
    chunk_sizes = adaptive_chunk_size(
        df_range['HA_close'],
        base_chunk=base_chunk_size,
        max_chunk=base_chunk_size*4,
        trend_threshold=0.001
    )

    # We'll store chunk start/end indices in this list
    chunk_indices = []

    idx = df_range.index[0]   # Start from the first datetime in df_range
    i   = 0                   # We'll move through the chunk_sizes array
    end_datetime = df_range.index[-1]

    while True:
        if i >= len(chunk_sizes):
            # If we've run out of chunk_sizes, break
            break
        current_chunk = chunk_sizes[i]
        s_time = idx
        row_pos = df_range.index.get_loc(s_time)
        e_pos = row_pos + current_chunk
        if e_pos >= len(df_range):
            e_pos = len(df_range) - 1  # Last valid position
        e_time = df_range.index[e_pos]
        chunk_indices.append((s_time, e_time))
        # Update for next iteration
        i = e_pos + 1  # move to next index in chunk_sizes
        if e_pos + 1 < len(df_range):
            idx = df_range.index[e_pos + 1]
        else:
            break
        if e_time == end_datetime:
            break
    checkpoint=yue_wang_modification_test(df_range['HA_close'].values).trend
    if checkpoint == 'increasing' or checkpoint=='decreasing':
        test_result = []
        test_result_filtered = []
        for (start_time, end_time) in chunk_indices:
            segment = df_range.loc[start_time:end_time, 'HA_close']
        # Apply filtering to that chunk
            segment_filtered = filter_signal(segment, segment.index)
        # Mann-Kendall on raw segment
            mk_raw = yue_wang_modification_test(segment.values)
            if mk_raw[0] == 'increasing':
                test_result.append(1)
            elif mk_raw[0] == 'decreasing':
                test_result.append(-1)
            else:
                test_result.append(0)
        # Mann-Kendall on filtered segment
            mk_filt = yue_wang_modification_test(segment_filtered.values)
            if mk_filt[0] == 'increasing':
                test_result_filtered.append(1)
            elif mk_filt[0] == 'decreasing':
                test_result_filtered.append(-1)
            else:
                test_result_filtered.append(0)
        df_filtered = filter_signal(df_range['HA_close'], df_range.index)
        detrended_segments = []
        trend_regions = []
        trend_lines   = []

    # We'll track each chunk's results here
        for idx_chunk, (start_time, end_time) in enumerate(chunk_indices):
            segment = df_range.loc[start_time:end_time, 'HA_close'].values
            trend_flag = test_result_filtered[idx_chunk]
            s_pos = df_range.index.get_loc(start_time)
            e_pos = df_range.index.get_loc(end_time)
            length = (e_pos - s_pos) + 1
            x_vals = np.arange(length).reshape(-1, 1)
            if trend_flag != 0:
                # Trend: use linear regression
                model = LinearRegression()
                model.fit(x_vals, segment.reshape(-1, 1))
                trend_line = model.predict(x_vals).flatten()
                detrended_seg = segment - trend_line
                trend_regions.append((s_pos + start_index, e_pos + start_index + 1, trend_flag))
                trend_lines.append((s_pos + start_index, e_pos + start_index + 1, trend_line))
            else:
                # Mean-reverting: subtract mean
                detrended_seg = segment - np.mean(segment)
            detrended_segments.append(detrended_seg)
        # Combine all detrended segments into a single 1D array
        detrended_values = np.concatenate(detrended_segments)
        # Detrended index
        detrended_index = df_range.index[:len(detrended_values)]
        detrended_full  = pd.Series(detrended_values, index=detrended_index)
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 6))

        # Plot original data
        #mpf.plot(df[start_time:end_time], type='candle', ax=ax1, style='yahoo', volume=False)
        #ax1.set_title("Asset Price with Trend")
        ax1.plot(df_range.index, df_range['HA_close'], label='Original', color='orange')
        # Detrended data
        ax2.plot(detrended_index, detrended_full, label='Detrended', color='blue')
        ax2.axhline(0, linestyle="dashed", color="black", alpha=0.7)
        ax2.legend()

        # Highlight trend regions on the top plot
        for (s_idx, e_idx, trend) in trend_regions:
            color = 'green' if trend == 1 else 'red'
            # Convert back to timestamps
            s_time = df.index[s_idx]
            e_time = df.index[e_idx - 1]  # e_idx is exclusive
            ax1.axvspan(s_time, e_time, color=color, alpha=0.2)

        # Trend lines
        for (s_idx, e_idx, line) in trend_lines:
            s_time = df.index[s_idx]
            e_time = df.index[e_idx - 1]
            # We need the time index for the chunk
            x_range = pd.date_range(start=s_time, periods=len(line), freq='15T')
            ax1.plot(x_range, line, color='black', linestyle='dashed', linewidth=1.5)

        ax1.legend()
        plt.tight_layout()
        plt.show()
        return detrended_full, checkpoint

    if checkpoint == 'no trend':
        print('The time series is not trending ')
        segment= df_range['HA_close']
        detrended_full = segment - segment.mean()

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 6))

        ax1.plot(df_range['HA_close'], label='Original', color='orange')

        # Detrended data
        ax2.plot(detrended_full, label='Detrended', color='blue')
        ax2.axhline(0, linestyle="dashed", color="black", alpha=0.7)
        ax2.legend()

        ax1.legend()
        plt.tight_layout()
        plt.show()
        
        return detrended_full, checkpoint
        
def rolling_window_detrend(series, window=50):
    detrended = np.zeros_like(series)
    for i in range(len(series)):
        start = max(0, i - window // 2)
        end = min(len(series), i + window // 2)
        x = np.arange(end - start)
        y = series[start:end]
        coeffs = np.polyfit(x, y, 1)  # Linear fit
        trend = np.polyval(coeffs, x)
        detrended[start:end] = y - trend  # Subtract trend
    return detrended
    
######
def find_period(x : np.array) -> float:
    """
    Calculate the periods between zero crossings in the given data.

    Parameters:
     - x (np.array): Array of corresponding data values.

    Returns:
     - np.array: Array of periods between zero crossings, each period is calculated as twice the difference between consecutive zero crossings.
    """
    from scipy.signal import savgol_filter
    x = savgol_filter(x, len(x)//10, 3, mode='nearest')
    zeros = find_zeros(x)
    periods = []
    for i in range(len(zeros)-1):
        periods.append((zeros[i+1] - zeros[i])*2)
    period = np.mean(periods)
    return period


def find_zeros(x : np.array) -> np.array:
    """
     Find the zeros of a function based on its sampled values.
    
     Parameters:
      - x (np.array): An array of dependent variable values corresponding to ⁠ t ⁠.

     Returns:
      - (np.array) An array of ⁠ t ⁠ values where ⁠ x ⁠ crosses zero.
     """
    t = np.arange(len(x))
    zeros = []
    for i in range(len(x)-1):
        if x[i]*x[i+1] < 0:
            zeros.append(t[i])
    return np.array(zeros, dtype=float)