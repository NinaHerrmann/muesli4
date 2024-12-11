//
// Created by Nina Herrmann on 11.12.24.
//

#ifndef DECISIONTREE_H
#define DECISIONTREE_H
#include <vector>

class DecisionTree {
public:
  // A simple decision stump (1-level decision tree)
  struct TreeNode {
    int featureIndex; // Index of the feature to split on
    double threshold; // Threshold value for splitting
    double leftValue; // Value assigned to left leaf
    double rightValue; // Value assigned to right leaf

    TreeNode(int featureIndex, double threshold, double leftValue, double rightValue)
        : featureIndex(featureIndex), threshold(threshold), leftValue(leftValue), rightValue(rightValue) {}
  };

  static TreeNode* buildStump(const std::vector<std::vector<double>>& data, const std::vector<double>& residuals) {
    // Parameters for the best split
    int bestFeatureIndex = -1;
    double bestThreshold = 0.0;
    double minError = std::numeric_limits<double>::max();
    double leftValue = 0.0, rightValue = 0.0;

    int numFeatures = data.size() - 1;

    // Iterate over all features
    for (int featureIndex = 1; featureIndex <= numFeatures; ++featureIndex) {
      // Extract feature values
      std::vector<double> featureValues;
      for (const auto& row : data) {
        featureValues.push_back(row[featureIndex]);
      }
      // Sorting feature values
      std::vector<double> uniqueValues = featureValues;
      std::sort(uniqueValues.begin(), uniqueValues.end());
      uniqueValues.erase(std::unique(uniqueValues.begin(), uniqueValues.end()), uniqueValues.end());

      // Checking each unique value as a threshold
      for (double threshold : uniqueValues) {
        double leftSum = 0.0, rightSum = 0.0;
        int leftCount = 0, rightCount = 0;

        for (size_t i = 0; i < data.size(); ++i) {
          if (featureValues[i] <= threshold) {
            leftSum += residuals[i];
            leftCount++;
          } else {
            rightSum += residuals[i];
            rightCount++;
          }
        }

        double leftValue = leftCount > 0 ? leftSum / leftCount : 0.0;
        double rightValue = rightCount > 0 ? rightSum / rightCount : 0.0;

        // Calculate error (squared loss)
        double error = 0.0;
        for (size_t i = 0; i < data.size(); ++i) {
          double prediction = featureValues[i] <= threshold ? leftValue : rightValue;
          error += (residuals[i] - prediction) * (residuals[i] - prediction);
        }
        if (error < minError) {
          minError = error;
          bestFeatureIndex = featureIndex;
          bestThreshold = threshold;
          leftValue = leftValue;
          rightValue = rightValue;
        }
      }
    }

    if (bestFeatureIndex == -1) return nullptr;

    return new TreeNode(bestFeatureIndex, bestThreshold, leftValue, rightValue);
  }
};

#endif //DECISIONTREE_H
