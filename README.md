[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/) [![CodeFactor](https://www.codefactor.io/repository/github/ityutin/df-and-order/badge)](https://www.codefactor.io/repository/github/ityutin/df-and-order) [![Maintainability](https://api.codeclimate.com/v1/badges/74ec941e646253e9e7ac/maintainability)](https://codeclimate.com/github/ityutin/df-and-order/maintainability) [![codecov](https://codecov.io/gh/ityutin/df-and-order/branch/master/graph/badge.svg)](https://codecov.io/gh/ityutin/df-and-order)

# 🗄️ df-and-order 
# Yeah, it's just like Law & Order, but Dataframe & Order!

```
pip install df_and_order
```

Using **df-and-order** your interactions with dataframes become very clean and predictable.

- Tired of absolute file paths to data in shared notebooks in your repository?
- Can't remember how your datasets were generated?
- Want to have safe and reproducible data transformations?
- Like declarative config-based solutions?

Good news for you!

## How it looks in code?
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

## Wow. Is it really magic?
**df-and-order** works with yaml configs. Every config contains metadata about a dataset as well as all desired transfomations.
Here's an example:
```yaml
df_id: user_activity_may_2020  # here's the dataframe identifier
initial_df_format: csv
metadata:  # this section contains some useful information about the dataset
  author: Data Man
  data_collection_date: 2020-05-01
transforms:
  model_input:  # here's the transform identifier
    df_format: csv
    in_memory:  # means we want to perform transformations in memory every time we calling it, permanent transforms are supported as well
    - module_path: df_and_order.steps.pd.DropColsTransformStep  # file where to find class describing some transformation. this one drops columns
      params:  # init params for the transformation class
        cols:
        - redundant_col
    - module_path: df_and_order.steps.DatesTransformStep  # another transformation that converts str to datetime
      params:
        cols:
        - date_col
```

## Okay, what exactly is a **df-and-order**'s transform?

Every transformation is about changing an initial dataset in any way.

A transformation is made of one or many steps. Each step represents some operation. 
Here are examples of such operations:
- dropping cols
- adding cols
- transforming existing cols
- etc

**df-and-order** uses subclasses of `DfTransformStepConfig` to describe a step. It's possible and highly recommended to declare init parameters for any step in config. 
Using Single Responsibility principle we achieve a granular control over our entire transformation.

Just by looking at the config you can say how the transformed dataframe was created.

[Take a look at the more detailed overview to find more exciting stuff.](https://github.com/ityutin/df-and-order/blob/master/examples/How-To.ipynb)

[I also wrote an article to describe the benefits, check it out! There are lemurs and stuff.](https://medium.com/@emmarrgghh/imagine-theres-no-mess-in-your-data-folder-859135bd1262)

Hope the lib will help somebody to boost the productivity.

