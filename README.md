1. (Readin file) Goto the following website to download the Breast Cancer Dataset. Read the CSV file into
panda data frame. You can choose any data structure type to store the values (pandas Or Mat-Lab).
32 features (or columns).
2. (data cleaning / Preparation) Remove any row which contain empty cell(s) ”Bad Data.” Split the
dataset into training set (80%) and testing (20%).
3. (1st team member, Decision Tree) Train a Decision Tree Classifier using the training set. Track the
training time using time module. <br>
(a) Draw the decision Tree. <br>
(b) Evaluate your trained model using the testing data. <br>
(c) How well does your model perform? Use performance metrics, like accuracy, sensitivity and
specificity (recall). <br>
(d) Visualize the confusion Matrix.
4. (1st team member, Random Forest Map-Reduce Ensemble) Use the training set and train all on three
decision trees (it should be implemented using Multiprocessing module). Each decision tree have
different max depth (max depth = 3, 5, and 7). Make predictions on the testing set using the trained
three decision trees. Combine each of the three predictions using majority voting (ensemble). Track
the training time using time module. <br>
(a) Draw each of the three decision Trees. <br>
(b) Evaluate your trained model using the testing data. Use performance metrics, like accuracy,
sensitivity and specificity (recall). <br>
(c) Visualize the confusion Matrix. <br>
(d) Compare the training times to question 3. <br>
5. (2nd team member, Support Vector Machine) Train the dataset on the Support Vector Machine (RBF)
Classifier (Track the training time). <br>
(a) visualize the class boundary using any two features using your trained SVM classifier. <br>
(b) Evaluate your trained model using the testing data. Use performance metrics, like accuracy,
sensitivity and specificity (recall). <br>
(c) Visualize the confusion Matrix. <br>
6. (2nd team member, SVM Map-Reduce Ensemble) Use the training set and train all on three SVM
Classifiers (it should be implemented using Multiprocessing module). Each SVM classifier must have
a different kernel (linear, RBF or Polynomial). Make predictions on the testing set using each of the
trained three Classifiers. Combine each of the three predictions using majority voting (ensemble).
Track the training time using time module. <br>
(a) visualize the class boundary using any two features using your trained SVM classifier. <br>
(b) Evaluate your trained model using the testing data. Use performance metrics, like accuracy,
sensitivity and specificity (recall). <br>
(c) Visualize the confusion Matrix. <br>
(d) Compare the training times to question 5.
