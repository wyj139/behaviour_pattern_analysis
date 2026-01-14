# Behavior Pattern Analysis Pipeline

A modular toolkit for analyzing animal behavior trajectory data using deep learning and statistical modeling.

## Overview

This pipeline processes trajectory data through five main stages:
1. Data preprocessing with coordinate transformation and smoothing
2. Sliding window segmentation of time series data
3. Feature extraction using a convolutional autoencoder
4. Behavior state classification with Gaussian Hidden Markov Models
5. Dimensionality reduction and visualization using UMAP
 
 The data is not being uploaded because it is too big
 
 Each of these steps can be skip if you already got the data, 
 if you need to skip any step, run the code as below:
    python main.py \
    --skip-preprocessing \
    --skip-training \
    --skip-visualization

## Project Structure

```
config.py              - Global configuration parameters
data_prep.py         - Data loading and preprocessing
windowing.py           - Sliding window segmentation
autoencoder.py         - Convolutional autoencoder model
hmm_classifier.py      - HMM-based state classification
visualization.py       - UMAP visualization tools
main.py               - Main pipeline orchestration
```

## Dependencies

pandas, numpy, torch, scikit-learn, hmmlearn, umap-learn, plotly, matplotlib

## Module Documentation

### config.py

Central configuration file containing all adjustable parameters:

**Data Paths:**
- RAW_META_CSV: Input metadata file path
- RAW_TIMESERIES_CSV: Input trajectory data file path
- OUTPUT_BASE_DIR: Base directory for outputs

**Window Parameters:**
- WINDOW_SIZE: Number of frames per window (default: 50)
- WINDOW_STRIDE: Step size for sliding window (default: 20)

**Autoencoder Parameters:**
- EMBEDDING_DIM: Dimensionality of latent embeddings (default: 32)
- CONV_CHANNELS: Number of convolutional channels (default: 32)
- BATCH_SIZE: Training batch size (default: 64)
- NUM_EPOCHS: Training epochs (default: 360)
- LEARNING_RATE: Optimizer learning rate (default: 1e-4)

**HMM Parameters:**
- N_STATES: Number of hidden states (default: 15, freely adjustable)
- HMM_COVARIANCE_TYPE: Covariance matrix type (default: "diag")

**UMAP Parameters:**
- UMAP_N_COMPONENTS: Output dimensionality (default: 2)
- UMAP_COLOR_OPTIONS: Available color-coding variables

**Visualization Skip Options:**
- SKIP_VIS_EMBEDDING_UMAP: Skip embedding-based UMAP plots
- SKIP_VIS_COORDINATE_UMAP: Skip coordinate-based UMAP plots
- SKIP_VIS_WINDOW_EMBEDDING_UMAP: Skip window embedding UMAP plots
- SKIP_VIS_WINDOW_COORDINATE_UMAP: Skip window coordinate UMAP plots

### data_prep.py

This module provides comprehensive data preparation utilities for behavioral trajectory analysis. It includes functions for reading metadata, generating unique sample IDs, filtering by multiple conditions, smoothing and imputing missing values, and calculating relative positions.

**Key Functions:**

- `generate_uid_column(df, uid_cols, uid_col_name, sep)`
  - Generates a unique string identifier for each row by concatenating specified columns.
  - Parameters:
    - df: Input DataFrame
    - uid_cols: List of columns to concatenate
    - uid_col_name: Name of the new UID column
    - sep: Separator string
  - Returns: DataFrame with new UID column

- `preprocess_dataframe_types(df)`
  - Standardizes data types for key columns (e.g., decision as int, others as string).
  - Parameters:
    - df: Input DataFrame
  - Returns: DataFrame with standardized types

- `filter_by_decision_cue_bodypart(df, decision, bodypart, pig=None, cue_type=None, base_outdir=OUTPUT_BASE_DIR)`
  - Filters the DataFrame by decision, bodypart, and optionally pig and cue type. Saves results to a subfolder.
  - Parameters:
    - df: Input DataFrame
    - decision: Decision value to match
    - bodypart: Bodypart(s) to match
    - pig: Optional list of pig IDs
    - cue_type: Optional list of cue types
    - base_outdir: Output base directory
  - Returns: Filtered DataFrame and folder label

- `calculate_relative_positions(df, reference_bodyparts=None, ...)`
  - Calculates relative x/y positions using the mean of specified reference bodyparts as the center for each frame.
  - Parameters:
    - df: Input DataFrame
    - reference_bodyparts: List of bodyparts to use as reference (default: all)
  - Returns: DataFrame with rel_x, rel_y, center_x, center_y columns

- `smooth_and_impute_all(df, x_col="x", y_col="y", frame_col="frame", ...)`
  - Smooths and imputes missing values for all coordinates and relative positions, grouped by sample and bodypart.
  - Parameters:
    - df: Input DataFrame
    - x_col, y_col: Coordinate column names
    - frame_col: Frame column name
    - max_speed: Maximum allowed movement per frame
    - calculate_relative: Whether to calculate relative positions
    - reference_bodyparts: Reference bodyparts for center calculation
  - Returns: DataFrame with smoothed and imputed values

- `prepare_meta_and_timeseries(meta, decision, bodypart, cue_type=None, pig=None, ...)`
  - High-level wrapper for the full data preparation process:
    1. Reads and processes metadata, generates UID
    2. Filters by decision, bodypart, cue, and pig
    3. Smooths and imputes time series data, calculates relative positions
    4. Saves results to subfolders
  - Parameters:
    - meta: Metadata DataFrame
    - decision: Decision value to filter
    - bodypart: Bodypart(s) to filter
    - cue_type: Optional cue types
    - pig: Optional pig IDs
    - uid_cols, uid_col_name: UID generation settings
    - out_base: Output base directory
    - calculate_relative: Whether to calculate relative positions
    - reference_bodyparts: Reference bodyparts for center calculation
  - Returns: Dictionary mapping folder label to saved file path

**Input Format:**
- DataFrame or CSV with columns: frame, bodypart, likelihood, x, y, id, pig, cue, decision, Housing, day, etc.

**Output:**
- Filtered and preprocessed CSV files saved to subfolders under the output directory, with all relevant columns and added rel_x, rel_y, center_x, center_y.

### windowing.py

Functions for sliding window segmentation.

**create_windows_by_bodypart(df, window_size, stride, output_dir)**
- Parameters:
  - df: Preprocessed DataFrame
  - window_size: Number of frames per window
  - stride: Step size between consecutive windows
  - output_dir: Directory for saving window data
- Returns: DataFrame with columns:
  - id: Sample identifier
  - window_id: Window index
  - all_window_rel_x: List of relative x-coordinates for all bodyparts
  - all_window_rel_y: List of relative y-coordinates for all bodyparts
  - bodyparts: List of bodypart names
  - n_bodyparts: Number of bodyparts
  - metadata columns (pig, cue, Housing, decision, etc.)
- Output: Saves window_data.csv

**load_window_data(file_path)**
- Parameters:
  - file_path: Path to saved window data CSV
- Returns: DataFrame with loaded window data

### autoencoder.py

Convolutional autoencoder for trajectory embedding.

**ConvAutoencoder**
- Architecture: Multi-layer 1D convolutional encoder-decoder
- Input shape: (batch_size, 2 * n_bodyparts, window_size)
- Output: Reconstructed trajectories with same shape as input
- Latent dimension: Configurable via EMBEDDING_DIM

**train_autoencoder(window_df, window_size, emb_dim, batch_size, num_epochs, learning_rate, output_dir)**
- Parameters:
  - window_df: Window data from windowing step
  - window_size: Frames per window
  - emb_dim: Embedding dimension
  - batch_size: Training batch size
  - num_epochs: Number of training epochs
  - learning_rate: Optimizer learning rate
  - output_dir: Directory for saving outputs
- Returns: Trained model and loss history
- Output: Saves autoencoder_model.pth and training_loss.png

**extract_embeddings(model, window_df, window_size, batch_size, output_dir)**
- Parameters:
  - model: Trained autoencoder model
  - window_df: Window data
  - window_size: Frames per window
  - batch_size: Batch size for inference
  - output_dir: Directory for saving embeddings
- Returns: DataFrame with embeddings and metadata
- Output: Saves embeddings.csv

### hmm_classifier.py

Gaussian Hidden Markov Model for state classification.

**train_gaussian_hmm(embedding_df, n_states, output_dir)**
- Parameters:
  - embedding_df: DataFrame with embeddings from autoencoder
  - n_states: Number of hidden states
  - output_dir: Directory for saving results
- Returns: 
  - window_with_states: DataFrame with state assignments
  - trans_matrix: State transition matrix
  - hmm_model: Trained HMM model
  - segments_df: DataFrame with state segments
- Output: Saves windows_with_states_n{N}.csv and state_segments_n{N}.csv

**analyze_state_transitions(trans_matrix, n_states, output_dir)**
- Parameters:
  - trans_matrix: Transition probability matrix
  - n_states: Number of states
  - output_dir: Directory for saving analysis
- Output: Saves transition_matrix_n{N}.csv and state_transition_analysis_n{N}.csv

**get_state_statistics(window_with_states, segments_df, output_dir)**
- Parameters:
  - window_with_states: DataFrame with state assignments
  - segments_df: State segments DataFrame
  - output_dir: Directory for saving statistics
- Output: Saves state_statistics_n{N}.csv

### visualization.py

UMAP-based dimensionality reduction and visualization.

**create_multiple_umaps(window_with_states, color_by_list, from_embeddings, output_dir, title_suffix)**
- Parameters:
  - window_with_states: DataFrame with states and embeddings or coordinates
  - color_by_list: List of column names for color coding
  - from_embeddings: Boolean, use embeddings (True) or coordinates (False)
  - output_dir: Directory for saving plots
  - title_suffix: Suffix added to plot titles
- Output: Saves interactive HTML plots for each color-by variable

**create_umap_from_coordinates(df, color_by, hover_cols, output_dir, title_suffix)**
- Parameters:
  - df: Preprocessed data with coordinates
  - color_by: Column name for color coding
  - hover_cols: List of columns to display on hover
  - output_dir: Directory for saving plot
  - title_suffix: Suffix for plot title
- Output: Saves umap_coords_{color_by}.html

**create_expanded_window_umap(window_df, embedding_df, window_size, window_id_filter, color_by, use_embeddings, output_dir)**
- Parameters:
  - window_df: Window data
  - embedding_df: Embeddings DataFrame (required if use_embeddings=True)
  - window_size: Frames per window
  - window_id_filter: Filter windows by ID (None for all windows)
  - color_by: Column name for color coding
  - use_embeddings: Boolean, use embeddings (True) or coordinates (False)
  - output_dir: Directory for saving plot
- Output: Saves umap_window{id}_emb_{color_by}.html or umap_window{id}_coords_{color_by}.html

**create_3d_trajectory_plot(traj_df, segments_df, target_id, bodypart, output_dir)**
- Parameters:
  - traj_df: Trajectory DataFrame
  - segments_df: State segments DataFrame
  - target_id: Sample ID to plot
  - bodypart: Specific bodypart to plot (None for all)
  - output_dir: Directory for saving plot
- Output: Saves 3D trajectory plot as HTML

### main.py

Pipeline orchestration and command-line interface.

**run_pipeline(...)**
Main function parameters:
- output_base_dir: Output directory path
- reference_bodyparts: List of bodypart names for center calculation
- window_size: Frames per window
- window_stride: Sliding step size
- emb_dim: Embedding dimension
- batch_size: Training batch size
- num_epochs: Training epochs
- learning_rate: Optimizer learning rate
- n_states: Number of HMM states
- color_options: List of variables for UMAP color coding
- skip_preprocessing: Skip data preprocessing step
- skip_windowing: Skip window segmentation step
- skip_training: Skip autoencoder training step
- skip_hmm: Skip HMM classification step
- skip_visualization: Skip all visualization steps
- skip_vis_embedding_umap: Skip embedding-based UMAP plots
- skip_vis_coordinate_umap: Skip coordinate-based UMAP plots
- skip_vis_window_embedding_umap: Skip window embedding UMAP plots
- skip_vis_window_coordinate_umap: Skip window coordinate UMAP plots

Returns dictionary with processed data at each stage.

## Input Data Format

Required CSV columns:
- frame: Frame number (integer)
- bodypart: Bodypart name (string)
- likelihood: Detection confidence (float, 0-1)
- x, y: Raw coordinates (float)
- id: Sample identifier (string)
- Metadata: pig, trial, cue, decision, Housing, day, duration, date, test

Optional columns (auto-calculated if missing):
- center_x, center_y: Reference center coordinates
- rel_x, rel_y: Coordinates relative to center

## Output Files

All outputs are saved to the specified output directory:

**Preprocessing:**
- preprocessed_data.csv: Data with relative coordinates

**Windowing:**
- window_data.csv: Segmented window data

**Autoencoder Training:**
- autoencoder_model.pth: Saved model weights
- training_loss.png: Training loss curve
- embeddings.csv: Extracted latent embeddings

**HMM Classification:**
- windows_with_states_n{N}.csv: Windows with state labels
- state_segments_n{N}.csv: Continuous state segments
- transition_matrix_n{N}.csv: State transition probabilities
- state_transition_analysis_n{N}.csv: Transition statistics
- state_statistics_n{N}.csv: Per-state summary statistics

**Visualization:**
- umap_emb_{variable}_n{N}.html: Embedding-based UMAP plots
- umap_coords_{variable}.html: Coordinate-based UMAP plots
- umap_window{id}_emb_{variable}.html: Window embedding UMAP
- umap_window{id}_coords_{variable}.html: Window coordinate UMAP

## Key Features

**Adaptive Architecture:** The autoencoder automatically adjusts input dimensions based on the number of bodyparts in the data.

**Multi-Channel Input:** Each window contains all bodyparts as separate channels with shape (batch, 2 * n_bodyparts, window_size).

**Flexible State Number:** HMM state count is fully configurable via parameters or config file.

**Granular Visualization Control:** Each visualization type can be independently enabled or disabled.

**Modular Design:** All components can be used independently or as part of the full pipeline.
