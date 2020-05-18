from df_and_order.df_transform_step import DfTransformStep

class DummyTransformStep(DfTransformStep):
    def transform(self, df):
        return df
