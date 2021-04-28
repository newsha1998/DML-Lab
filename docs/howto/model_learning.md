# Model Learning

The code has two different parts, the training and testing model.

## Train Model
For the training model, you should have a train data set that each feature type is an integer.

```bash
python3 <model_type.py> -T <train_file.csv> -M <model-name>
```

## Testing Model
For predicting the test data set, you should mention the trained model as an argument.
```bash
python3 <model_type.py> -P <test_file.csv> -M <model-name> -O <output-file>
```

Your output file will save in the output-file.
