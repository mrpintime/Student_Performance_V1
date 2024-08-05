# Student Performance Analysis in Secondary Education 

![image info](./assets/Banner_G3.jpg)

# Project Overview
This project aims to predict the final grade (G3) of students in secondary education based on a range of demographic, social, family-related features, and academic performance data (grades G1 and G2). Utilizing datasets from two Portuguese schools covering both Mathematics (mat) and Portuguese language (por) courses, we seek to identify key predictors of student success and provide actionable insights to educators and policymakers.

# Table of Contents
- [Usage](#usage)
- [Data Description](#data-description)
- [Methodology](#methodology)
- [Results](#results)
- [Business Applications](#potential-business-and-practical-applications)
- [Next Steps](#next-steps)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

# Usage
- You can easily clone the project and see the main jupyter file on your local machine by using following command:

```bash
# Cloning
git clone git@github.com:mrpintime/Student_Performance_V1.git
```
or

```bash 
git clone https://github.com/mrpintime/Student_Performance_V1.git
```

# Data Description
The datasets include student grades, demographic, social, and school-related features from two Portuguese schools, focusing on Mathematics and Portuguese language courses. Key attributes include student grades (G1, G2, G3), family background, study time, health status, and more, aiming to predict the final grade (G3).  
> **Important Notice:** I've merged the datasets for both the Mathematics and Portuguese language courses, so when using our models, they'll assume which course you're from—Mathematics or Portuguese—based on accurate data (the G1 and G2 scores) you provide. Consequently, the predicted score reflects the specific course your entered data relate to.

# Problem
The main problem here is find out that students progress from demographic and family relation data and personal features and turn it into Business Advantages

# Methodology

## Data Preprocessing

1. **Data Cleaning:** We began data cleaning by Checking Null, Duplicate and Wrong values within dataset.

2. **Normalization:** We normalized data when we want to use PCA Transformation and when we try to create Regression models. We need to normalize data in these approches otherwise we may get incorret or inaccurate answers.

3. **Target Variable:** Our Target Variable is G3 score which is in Numeric type 

> **Important Notice:** We did not transform Target Variable using Any Scaler here therefore output of our models will be in scale of your score 

> **Future apprache:**   
    1.**Add Course Feature:** For future approches we may want to expand our dataset by adding Course feature that show each student course. 

## Feature Extraction

1. **Statistical Features:** We use Median, Mean, Standard Deviation of our features with combination of Kernel Density Estimation Function to get sence of their Distribution and aslo get basic understanding about variation or spread and balanced location of each features 

2. **Correlation:** We Extract Pearson correlation Coeffitient for each pair of features. These features provide a basic understanding of the relation of features based of variation.
  
3. **PCA:** Given the high dimensional dataset that we will face at the end of feature Engineering process and multicollinearity that we have we use PCA to decrease dimension of our dataset to make the comlexity of our dataset much less and also we can see relation between number of PCA components and Ratio that we explained variation of dataset.

4. **Clustering:** We used advantagous of Clustering algorithm like Kmeans to identify potential clusters in our dataset and also create a new feature in feature Engineering process.

5. **Feature Engineering**: Given dataset we have in this section we tried to create new features using basic features that we have to use to explain our problem.

The combination of these preprocessing and feature extraction techniques was critical in preparing the dataset for subsequent machine learning models. They allowed us to capture the essential characteristics of Student relevent to its G3 Score and explaining its progress.


## Exploratory Data Analysis

In this section, we harness the capabilities of Python libraries such as Seaborn, Matplotlib, Numpy, and Pandas to deepen our understanding of the dataset. We outline our methodologies as follows:

1. **Univariate Analysis:**
   - **Plots:** We employ various plots, including bar plots and histograms, to gain a comprehensive understanding of feature distribution. These plots are organized within a single figure using the subplot method, allowing for simultaneous analysis and comparison.
   - **Descriptive Statistics:** Through descriptive analysis, we attain a nuanced comprehension of the variation within our categorical and numerical variables, assessing their scale and distribution. This includes the application of measures such as mode, mean, and other statistical features.
   - **Outlier Detection:** We identify potential outliers using a graphical approach, specifically through box plots. This method helps us pinpoint and evaluate outliers in our dataset.

2. **Multivariate Analysis:**
   - **Plots:** Our plots explore the correlations between features, examining how they change in relation to each other. We uncover intriguing patterns by comparing features across different dimensions, including student schools, through multidimensional plots.
   - **Clustering:** Clustering techniques, such as K-means and Mean Shift, uncover potentially hidden patterns by grouping data based on similarity. This approach is applied in both one-dimensional and multidimensional spaces, revealing insightful patterns and relationships.

## Machine Learning Techniques and Algorithms

In this project, we employed a variety of machine learning techniques to predict G3 score of students. The following is an overview of the key algorithms and approaches we used:

### 1. Supervised Learning Algorithms:

This section outlines the models we have utilized in our project, each selected for its unique ability to address specific aspects of our data and analysis objectives:

1. **Polynomial Regression:** A form of linear regression in which the relationship between the independent variable x and the dependent variable y is modeled as an nth degree polynomial. Polynomial regression is suitable for capturing the nonlinear relationship between variables.

2. **Lasso Regression (Least Absolute Shrinkage and Selection Operator):** An extension of linear regression that adds a regularization term to the loss function. This term penalizes large coefficients, helping in feature selection and preventing overfitting by producing simpler models that may perform better on new data.

3. **Ridge Regression:** Similar to lasso regression, ridge regression adds a different type of regularization to the loss function, penalizing the sum of the squares of the model coefficients. This approach is particularly useful for dealing with multicollinearity and for improving model predictions by reducing overfitting.

4. **ElasticNet Regression:** A hybrid of lasso and ridge regression methods, incorporating both of their regularization terms. ElasticNet is versatile and can be used when there's a need to balance between feature selection and regularization, making it effective for models with many features or when dealing with highly correlated data.

Each of these models was chosen for its strengths in handling various types of data characteristics and complexities, ensuring a robust approach to our supervised learning challenges. By leveraging these algorithms, we aim to build predictive models that are both accurate and interpretable, suitable for addressing the specific needs of our project.


### 2. Unsupervised Learning for Feature Extraction:

- **K-Means Clustering:** We employ the K-Means algorithm as a foundational clustering technique to generate new features and explore potential clustering patterns, whether within a single dimension (one-dimensional space) or across multiple dimensions (multidimensional space). This method aids in identifying inherent groupings in the dataset that may not be immediately apparent.

> - **To further enrich our project**, we plan to incorporate **Autoencoders** alongside other advanced unsupervised machine learning and deep learning clustering algorithms in future phases. Autoencoders, in particular, offer a powerful mechanism for feature extraction and dimensionality reduction, allowing us to capture more nuanced patterns in the data. These steps will enable us to leverage the full spectrum of unsupervised learning techniques to uncover complex structures and relationships within our dataset, enhancing the depth and breadth of our analysis.

### 3. Model Evaluation and Selection:

In this critical phase of our project, we rigorously assess the performance of our models to ensure their reliability and accuracy in making predictions. Our evaluation strategy employs a combination of metrics and techniques as follows:

- **Performance Metrics:**
  - **MSE (Mean Squared Error):** This metric quantifies the average squared difference between the estimated and actual values, offering insight into the model's accuracy. Lower MSE values indicate higher precision.
  - **R-squared (R²):** This metric assesses the proportion of variance in the dependent variable explained by the independent variables, providing a measure of the model's explanatory power. Higher R² values signify a more effective model.

- **Cross-Validation Technique:** To validate the model's generalizability, we employ cross-validation, which involves dividing the dataset into several parts, using some for training and others for testing. This technique helps ensure the model's performance is consistent and reliable across different data samples.


- **Distribution Plots for Predicted vs. Actual Values:** An essential part of our evaluation involves visualizing the distribution of predicted values against actual values from the test set. By plotting these distributions, we can visually assess how closely the predicted values align with the actual ones, which is crucial for understanding the model's predictive performance in real-world scenarios. These plots help identify any systematic biases or discrepancies between predicted and actual outcomes, allowing us to fine-tune our models for better accuracy and reliability.

This comprehensive approach, combining quantitative metrics with visual analyses, ensures a robust evaluation of our models. Distribution plots for predicted versus actual values play a key role in this process, offering intuitive insights into model performance and guiding improvements to achieve more accurate and reliable predictions.

> **Important Notice:** We also have an Extra subsection in the End of Fine Tune Traditional Models section, i encourage you to see it. 

### 4. Hyperparameter Tuning:

- **Randomized SearchCV:** This method is integral to our hyperparameter optimization strategy, enabling us to efficiently identify the most effective settings for our models. Unlike grid search methods that exhaustively test all possible parameter combinations, Randomized SearchCV selects values for each hyperparameter randomly from a defined distribution over a fixed number of iterations. This approach significantly reduces computation time and resources while still providing a high chance of finding optimal or near-optimal model configurations. By sampling a wide range of values, Randomized SearchCV can explore the parameter space more broadly and often uncovers better solutions that might be missed by a more deterministic search strategy. This method has proven invaluable in refining our models' performance by pinpointing the best hyperparameter values that contribute to more accurate and reliable predictions.

# Results


### Note on Interpretation of Results

As you delve into the results section, here are some key points to keep in mind that will aid in understanding the performance and evaluation metrics presented:

- **Crafted Dataset:** The term "crafted" refers to datasets that have undergone feature engineering. This process involves creating new features, selecting relevant features, and transforming existing features to improve model performance. All models labeled as "crafted" are trained on these enhanced datasets, benefiting from the refined input space.

- **Consistent Evaluation Framework:** To ensure a fair and consistent comparison, all models, with the exception of the Polynomial Multiple Linear Regression (PolyNomial_MLR), were trained on the same train dataset. This consistency extends to the evaluation phase, where both the Mean Squared Error (MSE) and R-squared (R²) metrics are calculated against the same test dataset. This approach guarantees that the differences in model performance are attributable to the model's architecture and the feature engineering, rather than variations in the data used.

- **Cross-Validation:** The cross-validation metrics provided (STD and MEAN) were computed across the entire crafted dataset and not on a split one. This method ensures that the reported cross-validation scores accurately reflect the model's ability to generalize to new data, providing a robust measure of performance across different subsets of the dataset.

- **Polynomial Model Exception:** The PolyNomial_MLR model is an exception in our methodology, as it faces challenges that are distinct from the other models. Its training approach and the resulting metrics should be considered with this context in mind, especially when comparing its performance to that of the other models.

These guidelines are intended to help readers better understand the methodology behind our model training, evaluation, and the basis for our conclusions. By considering these points, we believe you will gain a more comprehensive understanding of our project's outcomes and the effectiveness of various predictive modeling techniques in the context of predicting student performance.

---

Our project utilized several machine learning models to predict student performance in secondary education, employing a range of techniques including Principal Component Analysis (PCA) with Multiple Linear Regression (MLR), Polynomial Multiple Linear Regression, and various crafted models with regularization techniques like Lasso and Ridge regression. Below, we present a summary of the performance metrics for each model:

| Model                      | MSE        | RScore    | Cross_Validation_STD | Cross_Validation_MEAN |
|----------------------------|------------|-----------|----------------------|-----------------------|
| PCA_MLR                    | 0.846757   | 0.903379  | 0.027317             | 0.878862              |
| PolyNomial_MLR             | 51.267082  | -4.849945 | 148.223689           | -81.654985            |
| Crafted_MLR                | 0.705747   | 0.919469  | 0.028407             | 0.895758              |
| Crafted_LassoNoTuned       | 0.659191   | 0.924782  | 0.028541             | 0.898658              |
| Crafted_RidgeNoTuned       | 0.694132   | 0.920794  | 0.028254             | 0.895314              |
| Crafted_RidgeTuned         | 0.694139   | 0.920794  | 0.028253             | 0.895315              |
| Crafted_LassoTuned         | 0.661711   | 0.924494  | 0.028436             | 0.898719              |
| Crafted_ElasticNetTuned    | 0.659716   | 0.924722  | 0.028608             | 0.898956              |

### Key Findings:
- **Model Performance:** The models with regularization techniques, particularly the `Crafted_LassoNoTuned`, `Crafted_LassoTuned`, and `Crafted_ElasticNetTuned`, showed the best performance in terms of both MSE and RScore, indicating their effectiveness in predicting student performance.
- **Polynomial Regression Challenges:** The `PolyNomial_MLR` model demonstrated significant issues, with a negative RScore and extremely high MSE due to poor model fit to the data.
- **Stability and Reliability:** The cross-validation results, particularly the standard deviation (STD) and mean, highlight the stability and reliability of the crafted models and the PCA over polynomial approaches. The crafted models with tuning (`Crafted_LassoTuned` and `Crafted_ElasticNetTuned`) exhibit slightly better cross-validation mean scores, suggesting they are more generalized across different data splits.
- **PCA_MLR as a Baseline:** Despite its simplicity, the `PCA_MLR` model performed reasonably well, offering a solid baseline for comparison. However, the crafted models with regularization clearly surpassed it in terms of predictive accuracy and model robustness.
- **Consistency Across Models:** The standard deviation of cross-validation scores was relatively low for all models `except` the Polynomial MLR, indicating consistent performance across different subsets of the data.
- **Optimal Model Selection:** The `Crafted_LassoTuned` and `Crafted_ElasticNetTuned` models stand out as the most promising options for predicting student grades, owing to their **high RScore** and **low MSE**, combined with **stable cross-validation** results.
- **Feature Importance:** We can see some features like `paid`, `famrel`, `schoolsup` and `absence` have significant impact in specific direction on elastic net model and also we can see such as these features in Lasso, it would be better if industry make good impacts on student progress by using above features.  

### Conclusion:
The analysis underscores the importance of feature engineering and model selection in predictive modeling. The regularization techniques (Lasso, Ridge, ElasticNet) have proven particularly beneficial in handling the complexity and potential overfitting, thus enhancing model performance. These findings offer valuable insights into predicting student performance, suggesting a focused direction for future research and application in educational settings.


# Potential Business and Practical Applications

### 1. **Personalized Learning Programs**
- Educational institutions can use the model predictions to design personalized learning programs tailored to the needs and strengths of individual students. By identifying areas where a student might struggle, educators can provide targeted interventions, resources, and support to improve their learning outcomes.

### 2. **Early Intervention Systems**
- Schools and educational authorities can develop early warning systems that identify students at risk of underperforming or dropping out. These systems can trigger timely interventions to address academic and non-academic issues, helping students stay on track and achieve their educational goals.

### 3. **Curriculum Development and Optimization**
- Insights from the predictive models can inform curriculum developers and educators about which areas of study are most challenging for students. This information can be used to revise and optimize curricula, making them more effective and aligned with student needs.

### 4. **Resource Allocation and Planning**
- By predicting student performance across different subjects and demographics, educational institutions can make informed decisions about resource allocation. This includes assigning teachers and support staff where they are needed most, investing in materials and programs that have the highest impact, and planning for future enrollment and class sizes.

### 5. **Student Counseling and Career Guidance**
- Counselors and advisors can use the predictive insights to better understand students' academic strengths and weaknesses, guiding them in course selection, extracurricular activities, and even career planning. This can help students make choices that align with their skills and interests, potentially improving educational satisfaction and outcomes.

### 6. **Educational Policy and Research**
- Policymakers and researchers can analyze the factors influencing student performance to develop evidence-based educational policies. This might include policies aimed at reducing educational disparities, enhancing family and community engagement in education, and improving the overall quality of education.

### 7. **EdTech Solutions**
- Technology companies specializing in educational software and applications can integrate the predictive models into their products. This could include adaptive learning platforms that adjust content difficulty based on a student’s predicted performance, or analytics tools for teachers to monitor student progress and intervene when necessary.

### 8. **Parental Engagement Tools**
- Developers can create apps or platforms that allow parents to monitor their child’s predicted academic trajectory, offering practical suggestions for support and engagement. This can help bridge the gap between home and school, fostering a supportive environment for student achievement.

These applications demonstrate the wide-ranging impact that predictive modeling of student performance can have, not just within the educational sector, but also in technology, policy making, and community support systems. By leveraging the insights gained from this project, stakeholders can work together to create more responsive, effective, and personalized educational experiences.

# Next Steps
Here we listed some next future steps on this project 

1. **Expand the Dataset:** We try out to build better and more related features and also add new feature to dataset

2. **Enhance Model Accuracy by Advanced Algorithms:** Experiment with more sophisticated machine learning algorithms, such as deep learning models, to potentially improve prediction accuracy.

3. **User Interface for Stakeholders:** Create a user-friendly dashboard that visualizes predictions and insights, making them accessible to educators, students, and parents without requiring technical expertise.


# Contributing

## Contributing to Student Performance Analysis in Secondary Education 

I highly appreciate contributions and are excited to collaborate with the community on this data science project. Whether it's through data analysis, model improvement, documentation, or reporting issues, your input is valuable. Here’s how you can contribute:

1. **Fork the Repository:** Start by forking the project repository to your GitHub account. This creates a personal copy for you to work on.

2. **Clone the Forked Repository:** Clone the repository to your local machine. This allows you to make changes and test them locally.

   ```git
   git clone https://github.com/mrpintime/Student_Performance_V1.git
   ```

3. **Create a New Branch:** Create a new branch for your work. This keeps your changes organized and separate from the main branch.

   ```git
   git checkout -b feature-or-fix-branch-name
   ```

4. **Contribute Your Changes:**
   - **Data Analysis:** If you’re adding new analysis, ensure your code is well-documented and follows the project’s coding conventions. Include comments and README updates explaining your methodology.
   - **Model Improvement:** For changes to existing models, provide a clear explanation and any performance metrics or results to support the improvements.
   - **Data Contribution:** If contributing new data, ensure it is properly cleaned, formatted, and accompanied by a source description.

5. **Commit and Push Your Changes:** Commit your changes with a clear message describing the update. Push the changes to your forked repository.

   ```git
   git commit -m "Detailed description of changes"
   git push origin feature-or-fix-branch-name
   ```

6. **Create a Pull Request:** Go to your fork on GitHub and initiate a pull request. Fill out the PR template with all necessary details.

7. **Code Review and Discussion:** Wait for the project maintainer(that's me 😁 ) to review your PR. Be open to discussion and make any required updates.

### Reporting Issues

- Use the Issues tab to report problems or suggest enhancements.
- Be as specific as possible in your report. Include steps to reproduce the issue, along with any relevant data, code snippets, or error messages.

### General Guidelines

- Adhere to the project's coding and data handling standards.
- Update documentation and test cases for substantial changes.
- Keep your submissions focused and relevant to the project's goals.

Your contributions play a vital role in the success and improvement of Student Performance Analysis in Secondary Education. We look forward to your innovative ideas and collaborative efforts!

# License
- This project is licensed under the MIT License - see the `LICENSE.md` file for details.

# Contact
- Contact me through my Linkedin: https://www.linkedin.com/in/moein-zeidanlou

# Acknowledgments
- Credits to UCI Machine Learning Repository for providing the dataset.  
Dataset Link: https://archive.ics.uci.edu/ml/datasets/student+performance
