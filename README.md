# Low Birth Weight Prediction with Machine Learning Algorithms
Based on the data set, I predict the possibility of low birth weight of a child given the following set of features:

* Age of the Mother in Years
* Weight in Pounds at the Last Menstrual Period
* Race (1 = White, 2 = Black, 3 = Other) 
* Smoking Status During Pregnancy (1 = Yes, 0 = No)
* History of Premature Labor (0 = None, 1 = Yes)
* History of Hypertension (1 = Yes, 0 = No)
* Presence of Uterine Irritability (1 = Yes, 0 = No)

Using this feature set (x0, x1 .. x6), I have generated higher order features to get greater accuracy of prediction. I have used Logistic Regression algorithm along with python modules scipy and numpy for this purpose.
After training on the data-set, I predicted the Low Birth Weight (0 = No, 1 = Yes) for the same data-set and compared it against the actual values. With this I calculated the accuracy of my prediction.
With feature mapping and regularization the algorithm achieved an accuracy of 82.14%.

# Data-set
## Source of data
        Hosmer and Lemeshow (2000) Applied Logistic Regression: Second
        Edition.  These data are copyrighted by John Wiley & Sons Inc. and must
        be acknowledged and used accordingly.  Data were collected at Baystate
        Medical Center, Springfield, Massachusetts.
Also look at the data_description for more details.

# Credits
* Hosmer and Lemeshow for providing the data-set.

