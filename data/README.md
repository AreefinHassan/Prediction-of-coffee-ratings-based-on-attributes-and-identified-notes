# Data

Source: coffee ratings data with attributes and tasting notes.

| File | Rows (excl. header) | Contents |
|---|---|---|
| `X_train.csv` | 620 | Training features: `coffee_id`, `roaster`, `roast`, `origin`, `100g_USD`, `review` |
| `X_test.csv` | 267 | Test features, same columns as `X_train.csv` |
| `format.csv` | 267 | Submission template: a single `class` column |
| `result.csv` | 267 | Model predictions for the test set: `class` is 1 for outstanding coffee, 0 for average |

Note: the training labels (`y_train.csv`) are not included in this upload and will be added separately.
