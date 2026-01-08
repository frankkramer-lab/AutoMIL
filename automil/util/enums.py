"""
Project level enums for AutoMIL.
"""
from enum import Enum

from slideflow.mil.models import Attention_MIL, TransMIL
from slideflow.mil.models.bistro.transformer import \
    Attention as BistroTransformer


# Logging Levels
class LogLevel(Enum):
    INFO = 20
    DEBUG = 10
    WARNING = 30
    ERROR = 40

class ModelType(Enum):
    Attention_MIL     = Attention_MIL
    TransMIL          = TransMIL
    BistroTransformer = BistroTransformer

    @property
    def model_name(self) -> str:
        """The associated string name to pass to slideflow"""
        name_mapping = {
            Attention_MIL: "attention_mil",
            TransMIL: "transmil", 
            BistroTransformer: "bistro.transformer"
        }
        return name_mapping[self.value]
    
    @property
    def model_class(self):
        """The associated torch module"""
        return self.value

# Resolution Presets for extracting dataset tiles (specifies tile size and magnification level)
class RESOLUTION_PRESETS(Enum):
    Ultra_Low = (2_000, "5x")
    Low   = (1_000, "10x")
    High  = (299, "20x")
    Ultra = (224, "40x")

    @property
    def tile_px(self) -> int:
        """Tile size in pixels"""
        return self.value[0]
    
    @property
    def magnification(self) -> str:
        """Tile magnification level"""
        return self.value[1]