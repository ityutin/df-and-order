import os
import yaml
import pandas as pd
from typing import Optional, Dict, List, Any

from df_and_order.df_transform import DfTransformConfig
from df_and_order.df_config import DfConfig
from df_and_order.df_cache import DfCache
from df_and_order.df_transform_state import DfTransformState
from df_and_order.df_transform_step import DfTransformStepConfig, DfTransformStep
from df_and_order.helpers import FileInspector, get_type_from_module_path


class DfReader:
    """
    Reads a dataframe from the disk using a config file.
    Helps you to organize and document your dataframes
    for easy access and reproducibility.

    Parameters
    ----------
    dir_path: str
        Absolute path where your dataframe is supposed to be saved.
    format_to_cache_map: dict
        To be able to work with various formats DfReader must know
        how to read&save them. Provide a map where a format extension is used as a key
        and DfCache instance as a value. See DfCache class documentation for the details.
    """
    def __init__(self,
                 dir_path: str,
                 format_to_cache_map: Dict[str, DfCache]):
        self._dir_path = dir_path
        self._format_to_cache_map = format_to_cache_map

    def df_exists(self,
                  df_id: str,
                  transform_id: Optional[str] = None) -> bool:
        """
        Checks whether a dataframe file exists at the provided path.

        Parameters
        ----------
        df_id: str
            Unique identifier of your dataframe.
        transform_id: optional str
            If you want to check whether a transformed version of your dataframe
            is persisted on the disk you may pass its unique identifier that
            matches the one in the config file.

        Returns
        -------
        True if the dataframe exists, False otherwise
        """
        df_dir_path = self._df_dir_path(df_id=df_id)
        if not DfConfig.config_exists(dir_path=df_dir_path):
            return False

        df_config = self._get_config(df_id=df_id)
        df_path = self._df_path(df_config=df_config,
                                df_id=df_id,
                                transform_id=transform_id)
        return DfReader._is_file_exists(path=df_path)

    def _df_last_modified_ts(self,
                             df_id: str,
                             transform_id: Optional[str] = None) -> float:
        df_config = self._get_config(df_id=df_id)
        df_path = self._df_path(df_config=df_config,
                                df_id=df_id,
                                transform_id=transform_id)
        result = FileInspector.last_modified_date(file_path=df_path)
        return result

    @staticmethod
    def _is_file_exists(path: str):
        return os.path.exists(path=path)

    def create_df_config(self,
                         df_id: str,
                         initial_df_format: str,
                         metadata: Optional[dict] = None,
                         transform: Optional[DfTransformConfig] = None):
        """
        Just a forwarding to DfConfig method, see docs in DfConfig.
        """
        DfConfig.create_config(dir_path=self._df_dir_path(df_id=df_id),
                               df_id=df_id,
                               initial_df_format=initial_df_format,
                               metadata=metadata,
                               transform=transform)

    def register_transform(self,
                           df_id: str,
                           df_config: DfConfig,
                           transform: DfTransformConfig):
        """
        Forms a filename for the given dataframe and adds a new transform to the config file if possible.
        In general it's just a forwarding to DfConfig method, see docs in DfConfig.
        """
        filename = self.df_filename(df_config=df_config,
                                    df_id=df_id,
                                    transform=transform)
        df_config.register_transform(transform=transform,
                                     filename=filename)

    def read(self,
             df_id: str,
             transform_id: Optional[str] = None,
             transform: Optional[DfTransformConfig] = None,
             forced: bool = False) -> pd.DataFrame:
        """
        Reads a dataframe from the disk. If you want a transformed version of your dataframe,
        but it's still not persisted, it first creates it and then reads it into memory.

        Parameters
        ----------
        df_id: str
            Unique identifier of your dataframe.
        transform_id: optional str
            Unique identifier of the desired transform.
        transform: optional DfTransformConfig
            Object that describes all the transformation steps required.
        forced: bool
            If True, no safety checks will be performed,
            so you take the responsibility for data consistency.

        Returns
        -------

        """
        if transform_id and transform:
            raise AttributeError('Provide either transform_id or transform_config')

        df_config = self._get_config(df_id=df_id)
        self._update_transforms_state(df_id=df_id)

        if transform_id or transform:
            if not transform:
                transform = df_config.transforms_by(transform_id=transform_id)

            return self._read_transformed(df_id=df_id,
                                          transform=transform,
                                          df_config=df_config,
                                          forced=forced)
        else:
            return self._read_initial(df_id=df_id,
                                      df_config=df_config)

    def _read_initial(self,
                      df_id: str,
                      df_config: DfConfig) -> pd.DataFrame:
        """
        Reads the original dataframe from the disk
        """
        df_format = df_config.initial_df_format
        return self._read_df(df_id=df_id,
                             df_format=df_format,
                             df_config=df_config)

    def _read_transformed(self,
                          df_id: str,
                          transform: DfTransformConfig,
                          df_config: DfConfig,
                          forced: bool = False) -> pd.DataFrame:
        """
        Reads the transformed dataframe from the disk or creates it if needed.
        """
        transform_id = transform.transform_id

        self.register_transform(df_id=df_id,
                                df_config=df_config,
                                transform=transform)

        transformed_df_exists = self.df_exists(df_id=df_id,
                                               transform_id=transform_id)
        if transformed_df_exists:
            return self._try_to_read_cached_transform(df_id=df_id,
                                                      df_config=df_config,
                                                      forced=forced,
                                                      transform=transform)

        df = self._read_source_df_for_transform(df_id=df_id,
                                                df_config=df_config,
                                                forced=forced,
                                                transform=transform)
        df = self._apply_df_transforms(df_id=df_id,
                                       df=df,
                                       df_config=df_config,
                                       transform=transform)

        return df

    def _read_source_df_for_transform(self,
                                      df_id: str,
                                      df_config: DfConfig,
                                      forced: bool,
                                      transform: DfTransformConfig):
        if transform.source_id:
            source_transform = df_config.transforms_by(transform_id=transform.source_id)
            df = self._read_transformed(df_id=df_id,
                                        transform=source_transform,
                                        df_config=df_config,
                                        forced=forced)
        else:
            df = self._read_initial(df_id=df_id,
                                    df_config=df_config)
        return df

    def _try_to_read_cached_transform(self,
                                      df_id: str,
                                      df_config: DfConfig,
                                      forced: bool,
                                      transform: DfTransformConfig):
        if transform.source_id:
            source_transform = df_config.transforms_by(transform_id=transform.source_id)
            self._try_validate_transform(df_id=df_id,
                                         transform=source_transform,
                                         forced=forced,
                                         child_transform=transform)

        self._try_validate_transform(df_id=df_id,
                                     transform=transform,
                                     forced=forced)
        return self._read_cached_transformed(df_id=df_id,
                                             transform=transform,
                                             df_config=df_config,
                                             forced=forced)

    def _apply_df_transforms(self,
                             df_id: str,
                             df: pd.DataFrame,
                             df_config: DfConfig,
                             transform: DfTransformConfig):
        transform_id = transform.transform_id
        df_format = transform.df_format
        both_transform_types_are_present = transform.in_memory_steps and transform.permanent_steps
        df_shape_before = df.shape
        if transform.source_in_memory_steps:
            df = DfReader._apply_transform_steps(df=df,
                                                 steps=transform.source_in_memory_steps)
        if transform.in_memory_steps:
            df = DfReader._apply_transform_steps(df=df,
                                                 steps=transform.in_memory_steps)
            df_shape_after_in_memory = df.shape
            df_shape_has_changed = df_shape_before != df_shape_after_in_memory
            if both_transform_types_are_present and df_shape_has_changed:
                raise Exception(f"Error: A permanent transform is also present, hence your "
                                f"in-memory transform can't modify the shape of the initial df.")
        if transform.permanent_steps:
            df = DfReader._apply_transform_steps(df=df,
                                                 steps=transform.permanent_steps)

            df_path = self._df_path(df_config=df_config,
                                    df_id=df_id,
                                    transform_id=transform_id)
            df_cache = self._get_df_cache(df_format=df_format)
            df_cache.save(df=df, path=df_path)
            source_transform = None
            if transform.source_id:
                source_transform = df_config.transforms_by(transform_id=transform.source_id)
            state = DfTransformState(transform=transform,
                                     source_transform=source_transform)
            self._save_transform_state(df_id=df_id, state=state)

        return df

    def _save_transform_state(self, df_id: str, state: DfTransformState):
        transforms_state = self._transforms_state_dicts(df_id=df_id)
        transform_id = state.transform.transform_id
        assert transforms_state.get(transform_id) is None
        transforms_state[transform_id] = state.to_dict()[1]
        self._save_transforms_state_file(df_id=df_id, transforms_state=transforms_state)

    def _transforms_states(self, df_id: str) -> Dict[str, DfTransformState]:
        state_dicts = self._transforms_state_dicts(df_id=df_id)
        result = {}

        for transform_id, state_dict in state_dicts.items():
            state = DfTransformState.from_dict(transform_id=transform_id,
                                               state_dict=state_dict)
            result[transform_id] = state

        return result

    def _transforms_state_dicts(self, df_id: str) -> Dict[str, Any]:
        file_path = self._transforms_state_file_path(df_id=df_id)
        result_dict = {}
        if os.path.exists(file_path):
            with open(file_path) as config_file:
                result_dict = yaml.safe_load(config_file)

        return result_dict

    def _save_transforms_state_file(self, df_id: str, transforms_state: Dict[str, Any]):
        with open(self._transforms_state_file_path(df_id=df_id), 'w') as config_file:
            yaml.dump(transforms_state, config_file)

    def _transforms_state_file_path(self, df_id: str) -> str:
        path = self._df_dir_path(df_id=df_id)
        file_path = os.path.join(path, '.transforms_state.yaml')
        return file_path

    def _try_validate_transform(self,
                                df_id: str,
                                transform: DfTransformConfig,
                                forced: bool,
                                child_transform: Optional[DfTransformConfig] = None):
        if len(transform.permanent_steps) == 0:
            return

        if not self.df_exists(df_id=df_id, transform_id=transform.transform_id):
            # at first I wanted to throw an exception here, but if the parent transform
            # is no longer present on the disk, such fact shouldn't prevent you from using a
            # serialized child transform. It would be a problem if persisted parent transform
            # is incompatible with the one used to generate a child transform's file
            # P.S. at least I understand it...
            return

        if len(transform.permanent_steps) > 0:
            self._validate_ts(df_id=df_id, forced=forced, transform=transform)

        transforms_state = self._transforms_states(df_id=df_id)
        if child_transform:
            maybe_transform_state = transforms_state.get(child_transform.transform_id)
            saved_transform = maybe_transform_state.source_transform
            # in case of child transform we better check the source separately
            # e.g. a source config can match source_transform in a child's state
            # but the source file was generated using different config before
            self._try_validate_transform(df_id=df_id,
                                         transform=transform,
                                         forced=forced)
        else:
            maybe_transform_state = transforms_state.get(transform.transform_id)
            saved_transform = maybe_transform_state.transform
        if not maybe_transform_state:
            base_warning = f"{transform.transform_id} transform was " \
                           f"persisted with an unknown configuration"
            if forced:
                print(f"VERY IMPORTANT WARNING: {base_warning}, but "
                      f"reading it anyway because the operation was forced.")
                return
            else:
                raise Exception(f"{base_warning}, can't safely load it")

        valid = saved_transform.to_dict()[1] == transform.to_dict()[1]

        if not valid:
            base_warning = f"You've changed the df_config.yaml file of {transform.transform_id}" \
                           f" transform, so it's incompatible with the persisted version"
            if forced:
                print(f"VERY IMPORTANT WARNING: {base_warning}, but "
                      f"reading it anyway because the operation was forced.")
                return
            else:
                raise Exception(base_warning)

    def _update_transforms_state(self, df_id: str):
        transforms_state = self._transforms_state_dicts(df_id=df_id)

        transform_ids_to_delete = []
        for transform_id, transform_dict in transforms_state.items():
            transform = DfTransformConfig.from_dict(transform_id=transform_id,
                                                    transform_dict=transform_dict)
            df_config = self._get_config(df_id=df_id)
            filename = DfReader.df_filename(df_config=df_config,
                                            df_id=df_id,
                                            transform=transform)
            file_path = self._df_dir_path(df_id=df_id, filename=filename)
            if not DfReader._is_file_exists(path=file_path):
                transform_ids_to_delete.append(transform_id)

        for transform_id in transform_ids_to_delete:
            del transforms_state[transform_id]

        self._save_transforms_state_file(df_id=df_id,
                                         transforms_state=transforms_state)

    def _read_cached_transformed(self,
                                 df_id: str,
                                 transform: DfTransformConfig,
                                 df_config: DfConfig,
                                 forced: bool = False) -> pd.DataFrame:
        df_format = transform.df_format
        transform_id = transform.transform_id

        if transform.permanent_steps:
            self._validate_ts(df_id=df_id, forced=forced, transform=transform)

        df = self._read_df(df_config=df_config,
                           df_id=df_id,
                           df_format=df_format,
                           transform_id=transform_id)

        if transform.in_memory_steps:
            df = DfReader._apply_transform_steps(df=df,
                                                 steps=transform.in_memory_steps)

        return df

    def _validate_ts(self, df_id: str, forced: bool, transform: DfTransformConfig):
        # if the code of one of the steps was modified since the transformed dataframe
        # was cached - we need to stop and warn a user about the need of regenerating it
        df_last_modified_date = self._df_last_modified_ts(df_id=df_id,
                                                          transform_id=transform.transform_id)

        def _filter_func(step_config: DfTransformStepConfig):
            # built-in types override the method to provide true last modification date
            step_class = get_type_from_module_path(module_path=step_config.module_path)
            step_last_modified_ts = step_class.step_last_modified_ts(step_config=step_config)
            result = step_last_modified_ts > df_last_modified_date
            return result

        outdated_steps = list(filter(_filter_func, transform.permanent_steps))
        if len(outdated_steps) > 0:
            steps_module_paths = [step.module_path for step in outdated_steps]
            base_warning = f'{steps_module_paths} steps of {transform.transform_id} transform were changed since the df was generated'
            if forced:
                print(f"Warning: {base_warning}, reading it anyway because the operation was forced")
            else:
                raise Exception(f'{base_warning}, '
                                'delete the file and try again to regenerate the df.')

    @staticmethod
    def _apply_transform_steps(df: pd.DataFrame,
                               steps: List[DfTransformStepConfig]) -> pd.DataFrame:
        """
        Applies all the steps for a transformation on the given dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            Initial dataframe to perform transformations on.
        steps: list of DfTransformStepConfig
            List of objects that represent a step of the whole transformation.

        Returns
        -------
        pd.DataFrame, fully transformed initial dataframe
        """
        transform_steps = [DfTransformStep.build_transform(config=step_config) for step_config in steps]
        for transform in transform_steps:
            df = transform.transform(df=df)

        return df

    def _read_df(self,
                 df_id: str,
                 df_format: str,
                 df_config: DfConfig,
                 transform_id: Optional[str] = None) -> pd.DataFrame:
        """
        General method for reading a dataframe from the disk.

        Parameters
        ----------
        df_id: str
            Unique identifier of your dataframe.
        df_format: str
            Format in which dataframe was saved.
        df_config: DfConfig
            Interface for working with a serialized dataframe config file.
        transform_id: optional str
            Unique identifier of the desired transform.

        Returns
        -------
        pd.DataFrame, the requested dataframe
        """
        df_path = self._df_path(df_config=df_config,
                                df_id=df_id,
                                transform_id=transform_id)

        df_cache = self._get_df_cache(df_format=df_format)
        df = df_cache.load(path=df_path)

        return df

    def _df_path(self,
                 df_config: DfConfig,
                 df_id: str,
                 transform_id: Optional[str] = None) -> str:
        """
        Forms a path to the dataframe.

        Parameters
        ----------
        df_config: DfConfig
            Interface for working with a serialized dataframe config file.
        df_id: str
            Unique identifier of your dataframe.
        transform_id: optional str
            Unique identifier of the desired transform.

        Returns
        -------
        str, absolute path to the dataframe
        """
        transform = None
        if transform_id:
            transform = df_config.transforms_by(transform_id=transform_id)
        filename = DfReader.df_filename(df_config=df_config,
                                        df_id=df_id,
                                        transform=transform)
        result = self._df_dir_path(df_id=df_id, filename=filename)

        return result

    @staticmethod
    def df_filename(df_config: DfConfig,
                    df_id: str,
                    transform: Optional[DfTransformConfig] = None):
        """
        Forms a filename for the dataframe

        Parameters
        ----------
        df_config: DfConfig
            Interface for working with a serialized dataframe config file.
        df_id: str
            Unique identifier of your dataframe.
        transform: optional DfTransformConfig
            Config representing a transform

        Returns
        -------
        str, a filename for the dataframe.
        """
        if transform:
            return f'{transform.transform_id}_{df_id}.{transform.df_format}'

        return f'{df_id}.{df_config.initial_df_format}'

    def _df_dir_path(self, df_id: str, filename: Optional[str] = None) -> str:
        """
        Forms an absolute file for a directory where you can find your dataframe or any other file.

        Parameters
        ----------
        df_id: str
            Unique identifier of your dataframe.
        filename: optional str
            If you want to get a path for some particular file, provide its filename.

        Returns
        -------
        str, absolute path to the desired item
        """
        path = self._df_dir_path_for(dir_path=self._dir_path,
                                     df_id=df_id)

        if filename:
            path = os.path.join(path, filename)

        return path

    @staticmethod
    def _df_dir_path_for(dir_path: str,
                         df_id: str) -> str:
        return os.path.join(dir_path, df_id)

    def _get_config(self, df_id: str) -> DfConfig:
        """
        Gets config object for the given dataframe.

        Parameters
        ----------
        df_id: str
            Unique identifier of your dataframe.

        Returns
        -------
        DfConfig instance.
        """
        result = self._get_config_for(dir_path=self._df_dir_path(df_id=df_id),
                                      df_id=df_id)
        return result

    @staticmethod
    def _get_config_for(dir_path: str, df_id: str) -> DfConfig:
        """
        Gets config object for the given dataframe at the given path.

        Parameters
        ----------
        dir_path: str
            Absolute path to the dir when the dataframe is located.
        df_id: str
            Unique identifier of your dataframe.

        Returns
        -------

        """
        result = DfConfig(df_id=df_id,
                          dir_path=dir_path)
        return result

    def _get_df_cache(self, df_format: str) -> DfCache:
        """
        Gives proper DfCache subclass for the given data format

        Parameters
        ----------
        df_format: str
            Format extension you want to use to save/load a dataframe.

        Returns
        -------
        DfCache instance.
        """
        df_cache = self._format_to_cache_map.get(df_format)
        if not df_cache:
            raise ValueError(f'Unknown df_format, df_cache was not provided: {df_format}')
        return df_cache
