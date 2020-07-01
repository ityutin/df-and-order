from dataclasses import dataclass
from typing import Optional, Tuple

from df_and_order.df_transform import DfTransformConfig


TRANSFORM_STATE_SOURCE_KEY = "source_transform"


@dataclass
class DfTransformState:
    transform: DfTransformConfig
    source_transform: Optional[DfTransformConfig]

    def to_dict(self) -> Tuple[str, dict]:
        """
        Converts an instance to serializable dictionary

        Returns
        -------
        tuple of str and dict, transform_id and transform's dictionary representation.
        """
        transform_id, transform_dict = self.transform.to_dict()
        result = transform_dict

        if self.source_transform:
            _, source_dict = self.source_transform.to_dict()
            result[TRANSFORM_STATE_SOURCE_KEY] = source_dict

        return transform_id, result

    @staticmethod
    def from_dict(transform_id: str,
                  state_dict: dict):
        """
        Builds DfTransformState instance out of serialized dictionary.

        Parameters
        ----------
        transform_id: optional str
            Unique identifier of a transform.
        state_dict:
            Dictionary representation of DfTransformState

        Returns
        -------
        DfTransformConfig instance.
        """
        state_dict = state_dict.copy()
        source_dict = state_dict.get(TRANSFORM_STATE_SOURCE_KEY)
        source_transform = None
        if source_dict:
            del state_dict[TRANSFORM_STATE_SOURCE_KEY]

        transform_dict = state_dict
        transform = DfTransformConfig.from_dict(transform_id=transform_id,
                                                transform_dict=transform_dict)

        if source_dict:
            source_transform = DfTransformConfig.from_dict(transform_id=transform.source_id,
                                                           transform_dict=source_dict)
        result = DfTransformState(transform=transform,
                                  source_transform=source_transform)
        return result
