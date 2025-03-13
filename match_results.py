import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set font and resolve minus sign display issue
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# Load CSV file - using the updated CSV file directly
file_path = "results/all_tattoos_vgg_comparisons.csv"
data = pd.read_csv(file_path)

# Separate data into same and different folders based on the Folder column
data_same = data[data["Folder"] == "same"]
data_different = data[data["Folder"] == "different"]

print(f"Same folder data count: {len(data_same)}")
print(f"Different folder data count: {len(data_different)}")

# Define 'Matches' ranges including 300
specified_matches = [50, 100, 150, 200, 250, 300]

# Calculate data counts greater than each specified matches number (same folder)
cumulative_counts_same = [
    (data_same["Matches"] > match).sum() for match in specified_matches
]
cumulative_percentages_same = (
    np.round((np.array(cumulative_counts_same) / len(data_same)) * 100, 2)
    if len(data_same) > 0
    else np.zeros(len(specified_matches))
)

# Calculate data counts greater than each specified matches number (different folder)
cumulative_counts_different = [
    (data_different["Matches"] > match).sum() for match in specified_matches
]
cumulative_percentages_different = (
    np.round((np.array(cumulative_counts_different) / len(data_different)) * 100, 2)
    if len(data_different) > 0
    else np.zeros(len(specified_matches))
)

# Plot line graph
plt.figure(figsize=(10, 6))
plt.plot(
    specified_matches,
    cumulative_percentages_same,
    marker="o",
    linestyle="-",
    color="b",
    label="Same Folder",
)
plt.plot(
    specified_matches,
    cumulative_percentages_different,
    marker="o",
    linestyle="-",
    color="r",
    label="Different Folder",
)

plt.xlabel("Matches Count")
plt.ylabel("Percentage of Data Greater Than Count (%)")
plt.title(
    "Percentage of Data Greater Than Specified Matches Count (Same vs Different Folder)"
)
plt.xlim(50, 300)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()

# Save the image
output_dir = Path("v_test/results/verify_new_company_analysis")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "verify_new_company_matches_comparison.png", dpi=300)

# Show the plot
plt.show()

# Get total data counts
total_data_same = len(data_same)
total_data_different = len(data_different)

# Update data for both categories and convert to DataFrame format
output_data_same = {
    "Matches Count": specified_matches,
    "Data Count Greater Than": cumulative_counts_same,
    "Total Data": [total_data_same] * len(specified_matches),
    "Percentage Greater Than": cumulative_percentages_same,
}

output_data_different = {
    "Matches Count": specified_matches,
    "Data Count Greater Than": cumulative_counts_different,
    "Total Data": [total_data_different] * len(specified_matches),
    "Percentage Greater Than": cumulative_percentages_different,
}

# Create two DataFrames
df_same = pd.DataFrame(output_data_same)
df_different = pd.DataFrame(output_data_different)

# Save two CSV files
df_same.to_csv(
    output_dir / "verify_new_company_same_folder_analysis.csv",
    index=False,
    encoding="utf-8-sig",
)
df_different.to_csv(
    output_dir / "verify_new_company_different_folder_analysis.csv",
    index=False,
    encoding="utf-8-sig",
)


# Function to save DataFrame as image
def save_dataframe_as_image(df, path, title):
    fig, ax = plt.subplots(figsize=(8, len(df) * 0.4))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.title(title, pad=60)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# Save table images for both DataFrames
save_dataframe_as_image(
    df_same,
    output_dir / "verify_new_company_same_folder_analysis.png",
    "Same Folder Analysis",
)
save_dataframe_as_image(
    df_different,
    output_dir / "verify_new_company_different_folder_analysis.png",
    "Different Folder Analysis",
)
# Calculate basic statistics for matches
matches_stats = pd.DataFrame(
    {
        "Statistic": ["Mean", "Median", "Min", "Max", "Std Dev"],
        "Same Folder": [
            data_same["Matches"].mean() if len(data_same) > 0 else 0,
            data_same["Matches"].median() if len(data_same) > 0 else 0,
            data_same["Matches"].min() if len(data_same) > 0 else 0,
            data_same["Matches"].max() if len(data_same) > 0 else 0,
            data_same["Matches"].std() if len(data_same) > 0 else 0,
        ],
        "Different Folder": [
            data_different["Matches"].mean() if len(data_different) > 0 else 0,
            data_different["Matches"].median() if len(data_different) > 0 else 0,
            data_different["Matches"].min() if len(data_different) > 0 else 0,
            data_different["Matches"].max() if len(data_different) > 0 else 0,
            data_different["Matches"].std() if len(data_different) > 0 else 0,
        ],
    }
)

# Save statistics to CSV
matches_stats.to_csv(
    output_dir / "verify_new_company_matches_statistics.csv",
    index=False,
    encoding="utf-8-sig",
)

# Save statistics as image
save_dataframe_as_image(
    matches_stats,
    output_dir / "verify_new_company_matches_statistics.png",
    "Matches Statistics Comparison",
)

print(f"Analysis complete. Results saved in {output_dir}")
