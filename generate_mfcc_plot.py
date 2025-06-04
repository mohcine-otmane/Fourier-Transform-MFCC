import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import soundfile as sf
import argparse
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from matplotlib.widgets import Slider, Button
from scipy.signal import butter, filtfilt

def set_style():
    plt.style.use('dark_background')
    mpl.rcParams['figure.facecolor'] = '#1a1a1a'
    mpl.rcParams['axes.facecolor'] = '#2a2a2a'
    mpl.rcParams['axes.edgecolor'] = '#404040'
    mpl.rcParams['axes.labelcolor'] = 'white'
    mpl.rcParams['xtick.color'] = 'white'
    mpl.rcParams['ytick.color'] = 'white'
    mpl.rcParams['text.color'] = 'white'

def apply_noise_reduction(y, sr, noise_reduce_strength=0.1, low_cutoff=20, high_cutoff=20000):
    # Apply bandpass filter
    nyquist = sr / 2
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(4, [low, high], btype='band')
    y_filtered = filtfilt(b, a, y)
    
    # Spectral gating
    D = librosa.stft(y_filtered)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Estimate noise floor
    noise_floor = np.mean(S_db) + noise_reduce_strength * np.std(S_db)
    
    # Apply noise gate
    S_db_cleaned = np.where(S_db < noise_floor, noise_floor, S_db)
    
    # Convert back to time domain
    S_cleaned = librosa.db_to_amplitude(S_db_cleaned)
    y_cleaned = librosa.istft(S_cleaned)
    
    return y_cleaned

def load_audio(file_path, noise_reduce=True):
    y, sr = librosa.load(file_path)
    if noise_reduce:
        y = apply_noise_reduction(y, sr)
    return y, sr

def compute_mfcc(y, sr, n_mfcc=13, hop_length=512, n_fft=2048):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    return mfccs, delta_mfccs, delta2_mfccs

def plot_features(y, sr, mfccs, delta_mfccs, delta2_mfccs):
    set_style()
    
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 0.1])
    
    # Waveform
    ax1 = fig.add_subplot(gs[0])
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='#00ff9f')
    ax1.set_title('Waveform', fontsize=12, pad=10)
    ax1.grid(True, alpha=0.3)
    
    # MFCCs
    ax2 = fig.add_subplot(gs[1])
    img2 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax2, cmap='magma')
    ax2.set_title('MFCCs', fontsize=12, pad=10)
    plt.colorbar(img2, ax=ax2, format='%+2.0f dB')
    ax2.grid(True, alpha=0.3)
    
    # Delta MFCCs
    ax3 = fig.add_subplot(gs[2])
    img3 = librosa.display.specshow(delta_mfccs, sr=sr, x_axis='time', ax=ax3, cmap='viridis')
    ax3.set_title('Delta MFCCs', fontsize=12, pad=10)
    plt.colorbar(img3, ax=ax3, format='%+2.0f dB')
    ax3.grid(True, alpha=0.3)
    
    # Delta2 MFCCs
    ax4 = fig.add_subplot(gs[3])
    img4 = librosa.display.specshow(delta2_mfccs, sr=sr, x_axis='time', ax=ax4, cmap='plasma')
    ax4.set_title('Delta2 MFCCs', fontsize=12, pad=10)
    plt.colorbar(img4, ax=ax4, format='%+2.0f dB')
    ax4.grid(True, alpha=0.3)
    
    # Add slider for time window
    ax_slider = fig.add_subplot(gs[4])
    time_slider = Slider(
        ax=ax_slider,
        label='Time Window (s)',
        valmin=0.1,
        valmax=len(y)/sr,
        valinit=min(5, len(y)/sr),
        color='#00ff9f'
    )
    
    def update(val):
        window = time_slider.val
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim(0, window)
        fig.canvas.draw_idle()
    
    time_slider.on_changed(update)
    
    # Add reset button
    reset_ax = plt.axes([0.8, 0.01, 0.1, 0.04])
    reset_button = Button(reset_ax, 'Reset View', color='#2a2a2a', hovercolor='#404040')
    
    def reset(event):
        time_slider.reset()
    
    reset_button.on_clicked(reset)
    
    plt.tight_layout()
    plt.show()

def save_features(mfccs, delta_mfccs, delta2_mfccs, output_path):
    features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    np.save(output_path, features)

def main():
    parser = argparse.ArgumentParser(description='Audio Feature Extraction and Visualization')
    parser.add_argument('input_file', help='Path to input audio file')
    parser.add_argument('--output', '-o', help='Path to save features (optional)')
    parser.add_argument('--n_mfcc', type=int, default=13, help='Number of MFCCs to compute')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length for feature extraction')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT window size')
    parser.add_argument('--no-noise-reduce', action='store_true', help='Disable noise reduction')
    parser.add_argument('--noise-strength', type=float, default=0.1, help='Noise reduction strength (0.0 to 1.0)')
    parser.add_argument('--low-cutoff', type=int, default=20, help='Low frequency cutoff in Hz')
    parser.add_argument('--high-cutoff', type=int, default=20000, help='High frequency cutoff in Hz')
    
    args = parser.parse_args()
    
    try:
        y, sr = load_audio(args.input_file, not args.no_noise_reduce)
        mfccs, delta_mfccs, delta2_mfccs = compute_mfcc(y, sr, args.n_mfcc, args.hop_length, args.n_fft)
        plot_features(y, sr, mfccs, delta_mfccs, delta2_mfccs)
        
        if args.output:
            save_features(mfccs, delta_mfccs, delta2_mfccs, args.output)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 