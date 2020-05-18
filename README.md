# 🗄️ df-and-order 
# Yeah, it's just like Law & Order, but Dataframe & Order!

```
pip install df_and_order
```

Using **df-and-order** your interactions with dataframes become very clean and predictable.

- Tired of absolute file paths in the shared notebooks in your repository?
- Can't remember how your dataframes were generated?
- Want to have a reproducibility on data transformations?
- Like declarative config-based solutions?

Good news for you!

Imagine the world where all you need to do for reading some dataframe you need just a few lines:

```python
reader = MagicDfReader()
df = reader.read(df_id='user_activity_may_2020')
```

Maybe you are interested in some transformed version of that dataframe? No problem!

```python
reader = MagicDfReader()
# ready to fit a model on!
model_input_df = reader.read(df_id='user_activity_may_2020', transform_id='model_input')
```

It is possible by having a config file that will look like this:
```yaml
df_id: user_activity_may_2020 # here's the dataframe identifier
initial_df_format: csv
metadata: # some useful information about the dataset
  author: Data Man
  data_collection_date: 2020-05-01
transformed_df_format: csv
transforms:
  model_input: # here's the transform identifier
    in_memory: # means we want to perform transformations in memory every time we calling it, permanent transforms are supported as well
    - module_path: df_and_order.steps.DropColsTransformStep # file with the transformation's code
      params: # init params for the transformation
        cols:
        - redundant_col
    - module_path: df_and_order.steps.DatesTransformStep - another transformation
      params:
        cols:
        - date_col
```

Just by looking at the config you can say how the transformed dataframe was created.

[Take a look at the more detailed overview to find more exciting stuff.](https://github.com/ityutin/df-and-order/blob/master/examples/How-To.ipynb)

Hope the lib will help somebody to boost the productivity.

Unit-tested, btw!

### Requirements
```
pandas
python 3.7
``` 
