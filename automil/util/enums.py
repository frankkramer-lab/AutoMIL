#==============================================================================#
#  AutoMIL - Automated Machine Learning for Image Classification in            #
#  Whole-Slide Imaging with Multiple Instance Learning                         #
#                                                                              #
#  Copyright (C) 2026 Jonas Waibel                                             #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.      #
#==============================================================================#
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