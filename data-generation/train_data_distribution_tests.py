import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def main():
    # Set plotting style
    sns.set_theme(style="whitegrid")

    # Load the train data
    print("Loading train_fleet_data.csv...")
    try:
        df = pd.read_csv('train_fleet_data.csv')
    except FileNotFoundError:
        print("Error: train_fleet_data.csv not found.")
        return
        
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    # Select all numerical columns to visualize
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Exclude ID columns and fault columns
    exclude_cols = ['Engine_number', 'trip_id']
    numerical_cols = [
        col for col in numerical_cols 
        if col not in exclude_cols and 'fault' not in col.lower()
    ]

    print(f"Plotting distributions for {len(numerical_cols)} numerical columns...")

    # Set up a grid layout
    n_cols = 4
    n_rows = math.ceil(len(numerical_cols) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for i, param in enumerate(numerical_cols):
        sns.kdeplot(data=df, x=param, fill=True, ax=axes[i], color='cornflowerblue')
        axes[i].set_title(f'Distribution of {param}')
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('Frequency / Density')

    # Hide any unused subplots
    for j in range(len(numerical_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    
    # Save the figure as an image
    output_filename = 'train_data_distributions.png'
    plt.savefig(output_filename, dpi=100)
    print(f"Distributions saved successfully to '{output_filename}'.")
    
    # Display the plots
    plt.show()

if __name__ == "__main__":
    main()
