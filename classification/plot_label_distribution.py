import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/UpdatedResumeDataSet.csv").dropna()

# Plot
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y="Category", order=df["Category"].value_counts().index, palette="Set2")
plt.title("Resume Category Distribution")
plt.xlabel("Count")
plt.ylabel("Category")

# Save plot
plt.tight_layout()
plt.savefig("classification/label_distribution.png")
print("✅ Saved to classification/label_distribution.png")
