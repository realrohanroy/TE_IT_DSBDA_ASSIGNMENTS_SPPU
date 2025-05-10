4. Python Operations on Facebook Metrics Dataset
1️⃣ How to load Facebook metrics data using pandas?

python
Copy
Edit
import pandas as pd
df = pd.read_csv('facebook_metrics.csv')
Explanation:
This reads a CSV file into a DataFrame df using pandas.

2️⃣ How do you create subsets using filter conditions?

python
Copy
Edit
high_likes = df[df['like'] > 500]
Explanation:
This selects rows where likes are more than 500.

3️⃣ How do you merge multiple dataframes?

python
Copy
Edit
merged_df = pd.merge(df1, df2, on='id')
Explanation:
Merges df1 and df2 based on common ‘id’ column.

4️⃣ What are the steps to sort a DataFrame?

python
Copy
Edit
sorted_df = df.sort_values(by='like', ascending=False)
Explanation:
Sorts data based on ‘like’ column in descending order.

5️⃣ How do you transpose a DataFrame?

python
Copy
Edit
df_transpose = df.T
Explanation:
Swaps rows with columns.

6️⃣ How do you reshape data using melt/pivot?

Melt:

python
Copy
Edit
df_melted = pd.melt(df, id_vars=['Page'], value_vars=['like', 'share'])
Explanation:
Converts wide data into long format.

Pivot:

python
Copy
Edit
df_pivot = df.pivot(index='Page', columns='Category', values='like')
Explanation:
Converts long data into wide format.

7️⃣ How do you handle missing values?

Check nulls: df.isnull().sum()

Drop: df.dropna()

Fill: df.fillna(0)

8️⃣ What is the use of groupby() in summarizing?

python
Copy
Edit
df.groupby('Category')['like'].mean()
Explanation:
Groups data by ‘Category’ and calculates average likes.

9️⃣ How to calculate correlation among features?

python
Copy
Edit
df.corr()
Explanation:
Calculates pairwise correlation between columns.

🔟 How to export the final processed data?

python
Copy
Edit
df.to_csv('processed_data.csv', index=False)
Explanation:
Saves DataFrame as a CSV file.

📌 5. Python on Air Quality and Heart Disease Data
1️⃣ How do you clean missing and invalid values?
Check: df.isnull().sum()
Remove: df.dropna()
Fill: df.fillna(method='ffill')

2️⃣ How do you integrate two different datasets?
Using pd.merge() or pd.concat()

3️⃣ How do you apply transformations like scaling?

python
Copy
Edit
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
Explanation:
Standardizes data to mean=0, variance=1.

4️⃣ How do you detect and correct outliers or errors?
Using IQR:

python
Copy
Edit
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['value'] < (Q1 - 1.5*IQR)) | (df['value'] > (Q3 + 1.5*IQR))]
5️⃣ How do you split data into train/test sets?

python
Copy
Edit
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
6️⃣ What models are suitable for heart disease prediction?

Logistic Regression

Decision Tree

Random Forest

SVM

KNN

7️⃣ How to evaluate model accuracy?

python
Copy
Edit
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
8️⃣ How to handle imbalanced data?

Oversampling (SMOTE)

Undersampling

Adjusting class weights

9️⃣ How to encode categorical features?

python
Copy
Edit
pd.get_dummies(df['category'])
or

python
Copy
Edit
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])
🔟 How to save and load models using joblib/pickle?

python
Copy
Edit
import joblib
joblib.dump(model, 'model.pkl')
loaded_model = joblib.load('model.pkl')
📌 6. Python Visualization (Matplotlib, Seaborn)
1️⃣ How to visualize air quality over time?

python
Copy
Edit
plt.plot(df['date'], df['AQI'])
plt.show()
2️⃣ What seaborn plot is suitable for correlation heatmaps?

python
Copy
Edit
sns.heatmap(df.corr(), annot=True)
3️⃣ How do you use subplots to show multiple variables?

python
Copy
Edit
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(df['AQI'])
4️⃣ How to plot boxplots for comparing groups?

python
Copy
Edit
sns.boxplot(x='city', y='AQI', data=df)
5️⃣ What is the best way to show trends using line graphs?
Using plt.plot() or sns.lineplot()

6️⃣ How do you add titles and labels to plots?

python
Copy
Edit
plt.title("Air Quality Trend")
plt.xlabel("Date")
plt.ylabel("AQI")
7️⃣ How to annotate values on plots?

python
Copy
Edit
plt.annotate('Highest AQI', xy=(10, 120), xytext=(12, 130))
8️⃣ How to customize figure aesthetics?

python
Copy
Edit
sns.set_style('whitegrid')
9️⃣ How to combine multiple plots into one figure?
Using plt.subplot() or plt.subplots()

🔟 How to save figures as PNG/PDF?

python
Copy
Edit
plt.savefig('plot.png')
📌 7. Tableau Visualizations on Adult & Iris Dataset
1️⃣ What is 1D visualization and how do you do it in Tableau?
A bar chart or histogram using one dimension.

2️⃣ How to create 2D scatter plots in Tableau?
Drag two continuous variables on X and Y axes.

3️⃣ Can Tableau represent 3D data?
Not directly — can simulate using multiple 2D charts or color/size for the third dimension.

4️⃣ How to create a time-series (temporal) chart?
Use Date field in Columns and measure in Rows.

5️⃣ How to handle multiple dimensions in a single view?
Use Rows, Columns, Color, Size, and Filters shelves.

6️⃣ How to build hierarchical (tree) diagrams in Tableau?
Create hierarchy by right-clicking on fields → Create Hierarchy.

7️⃣ How to visualize network data in Tableau?
Not native — external tools like Gephi are better.

8️⃣ How to filter and drill down data interactively?
Use filters, parameters, and dashboards actions.

9️⃣ How do you blend data from different sources?
By linking common fields in Data → Relationships.

🔟 How do you publish and share Tableau dashboards?
Use Tableau Public, Tableau Server, or export as PDF/Image.

📌 8. Python E-commerce Review Scraper
1️⃣ Which site are you targeting for scraping?
Example: Flipkart, Amazon, or custom.

2️⃣ How to inspect elements to find review data?
Right-click → Inspect → Check class/id of review tags.

3️⃣ How to handle dynamic loading (AJAX/JS) using Selenium?

python
Copy
Edit
from selenium import webdriver
driver = webdriver.Chrome()
driver.get(url)
4️⃣ What libraries are used for scraping?

requests

BeautifulSoup

Selenium

5️⃣ How to avoid getting blocked?
Use headers, sleep time, and proxies.

6️⃣ What fields are you extracting?
Name, Comment, Rating, Date.

7️⃣ How do you store data?
CSV, JSON, SQLite.

8️⃣ How to scrape multiple pages automatically?
Loop over page numbers in URL or pagination button clicks.

9️⃣ How to detect and remove duplicates?

python
Copy
Edit
df.drop_duplicates()
🔟 How to analyze and visualize scraped reviews?
Use word clouds, sentiment analysis, bar plots.

