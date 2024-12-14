Life Expectancy Prediction Using Machine Learning
=================================================

This project aims to predict the life expectancy of countries using various machine learning models. The dataset utilized in this project comes from the World Health Organization (WHO), and the focus is to compare the performance of different models to see which one yields the best results.

Overview
--------

The main goal of this project was to predict life expectancy based on a variety of factors such as income, education, healthcare, and other socio-economic factors using machine learning algorithms. Initially, I experimented with deep neural networks (DNN), but faced challenges with the model performance. Afterward, I explored other models like Linear Regression, Random Forests, and Gradient Boosting to find more reliable predictions.

### Key Steps

1.  Dataset: The project uses the WHO dataset, which contains various health and socio-economic factors for different countries, including life expectancy, income levels, literacy rates, etc.

2.  Modeling Approach:

    -   Deep Neural Network: I began by building a deep neural network, but it consistently got stuck at a validation loss of around 6.5. Although this was promising, the model did not reach a satisfactory performance.
    -   Linear Regression (via scikit-learn): I then visualized the data and observed a linear relationship between the features and the target variable (life expectancy). Using the pre-built `LinearRegression` function from scikit-learn, I achieved a mean absolute error (MAE) of 1.07.
    -   Random Forest and Gradient Boosting: I further experimented with scikit-learn tools, such as Random Forests and Gradient Boosting Regressor, but the results were quite similar to the Linear Regression model.
3.  Results and Graphs: After evaluating the models, I visualized the predictions of each model and compared them to the actual values, which provided insights into their accuracy and general performance.

* * * * *

Models Used
-----------

-   Deep Neural Network (DNN): A multi-layer neural network built using TensorFlow/Keras. Despite its potential, the model struggled to improve beyond a `val_loss` of 6.5.

-   Linear Regression: A simple but effective model from scikit-learn that produced a MAE of 1.07.

-   Random Forest Regressor: Another scikit-learn model that uses multiple decision trees to improve predictions.

-   Gradient Boosting Regressor: A powerful ensemble model known for building a strong predictive model from weak learners from scikit-learn.

* * * * *

Results
-------

-   Linear Regression (sklearn): MAE = 1.07
-   Random Forest Regressor: Similar performance to Linear Regression
-   Gradient Boosting Regressor: Comparable results to Random Forest and Linear Regression

The predictions from these models were plotted to visually compare them with the actual values, helping to confirm that the Linear Regression model performed the best.
![output](https://github.com/user-attachments/assets/6dec6694-0db0-4463-b911-6bf6bff02802)


* * * * *

Challenges
----------

-   Deep Neural Network Issues: Initially, I focused on building a deep neural network, but encountered performance issues where the validation loss plateaued around 6.5. Despite trying various hyperparameters, I was unable to significantly improve the results.

-   Model Convergence: Although Linear Regression and ensemble models performed better, it was challenging to further improve performance beyond a certain point.

* * * * *

Future Work
-----------

-   Hyperparameter Tuning: I plan to further explore the hyperparameter tuning for the DNN to see if it can be optimized for better performance.

-   Advanced Ensemble Methods: Investigating more advanced ensemble models, such as XGBoost or LightGBM, could lead to improved results.

-   Feature Engineering: Additional data preprocessing and feature engineering could be explored to improve model performance.

* * * * *

Technologies Used
-----------------

-   Python
-   TensorFlow/Keras for deep neural networks
-   scikit-learn for Linear Regression, Random Forests, and Gradient Boosting
-   Matplotlib and Seaborn for data visualization

* * * * *


Conclusion
----------

The project demonstrates the comparison of several machine learning models for life expectancy prediction. While the deep neural network showed promise, simpler models like Linear Regression performed surprisingly well, with an MAE of 1.07. Further work can be done to explore more advanced models and fine-tune them to improve prediction accuracy. Also, in the future I will try avoid using these prebuilt models as it's taking away from my learning, instaed I will persevere with hyperparameter tuning and model experimenting.

* * * * *

### 🚀 Feel free to contribute or open issues if you have any questions or suggestions! 🎉

* * * * *

Let me know if you need any more details or modifications!
