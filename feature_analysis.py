import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv("G2.csv")

# Clean column names
data.columns = data.columns.str.strip()

# Numeric features
numeric_features = [
    "height_dm",
    "weight_hg",
    "base_experience",
    "hp",
    "attack",
    "defense"
]

# =====================================================
# 1. Type with the highest average attack
# =====================================================

average_attack = (
    data.groupby("type_1")["attack"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
)

print("Average attack by primary type:")
print(average_attack)

highest_average_attack_type = average_attack["mean"].idxmax()
highest_average_attack_value = average_attack["mean"].max()

print(
    f"\nPrimary type with the highest average attack: "
    f"{highest_average_attack_type}"
)
print(f"Average attack: {highest_average_attack_value:.2f}")

# Plot average attack by type
plt.figure(figsize=(12, 6))

sns.barplot(
    data=data,
    x="type_1",
    y="attack",
    estimator="mean",
    errorbar=None,
    order=average_attack.index
)

plt.title("Average Attack by Primary Pokémon Type")
plt.xlabel("Primary Type")
plt.ylabel("Average Attack")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# If you meant the individual Pokémon with the highest attack:
highest_attack_pokemon = data.loc[data["attack"].idxmax()]

print("\nPokémon with the highest individual attack:")
print(highest_attack_pokemon[["id", "name", "type_1", "attack"]])


# =====================================================
# 2. Correlation between height and weight
# =====================================================

height_weight_correlation = data[["height_dm", "weight_hg"]].corr()

print("\nHeight-weight correlation:")
print(height_weight_correlation)

print(
    f"\nCorrelation coefficient: "
    f"{height_weight_correlation.loc['height_dm', 'weight_hg']:.3f}"
)

# Heatmap for height and weight
plt.figure(figsize=(5, 4))

sns.heatmap(
    height_weight_correlation,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5
)

plt.title("Correlation Between Height and Weight")
plt.tight_layout()
plt.show()


# Optional: scatterplot showing the relationship
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=data,
    x="height_dm",
    y="weight_hg",
    hue="type_1",
    alpha=0.7,
    palette="tab20"
)

plt.title("Height vs Weight")
plt.xlabel("Height (decimeters)")
plt.ylabel("Weight (hectograms)")
plt.legend(
    title="Primary Type",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)
plt.tight_layout()
plt.show()


# =====================================================
# 3. Correlation of numeric features with target type
# =====================================================

# Group means help show differences between types
group_means = data.groupby("type_1")[numeric_features].mean()

print("\nMean numeric features by primary type:")
print(group_means.round(2))

# Plot feature means by type
plt.figure(figsize=(14, 8))

sns.heatmap(
    group_means.T,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    linewidths=0.3
)

plt.title("Mean Numeric Features by Primary Pokémon Type")
plt.xlabel("Primary Type")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# =====================================================
# 4. Mutual information for feature usefulness
# =====================================================

# Remove rows with missing values in the selected columns
analysis_data = data[numeric_features + ["type_1"]].dropna()

X = analysis_data[numeric_features]
y = LabelEncoder().fit_transform(analysis_data["type_1"])

mutual_information = mutual_info_classif(
    X,
    y,
    random_state=42
)

feature_importance = (
    pd.DataFrame({
        "feature": numeric_features,
        "mutual_information": mutual_information
    })
    .sort_values("mutual_information", ascending=False)
)

print("\nFeature usefulness based on mutual information:")
print(feature_importance)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=feature_importance,
    x="mutual_information",
    y="feature",
    errorbar=None
)

plt.title("Mutual Information with Primary Pokémon Type")
plt.xlabel("Mutual Information")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()