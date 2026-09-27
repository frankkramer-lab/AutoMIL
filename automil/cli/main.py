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
"""The entry point CLI for running AutoMIL"""
# === External libraries === #
import warnings
# Suppressing warnings related to pkg_ressources and timm
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import click

# === Internal imports === #
from .commands.run_pipeline import run_pipeline
from .commands.train import train
from .commands.predict import predict
from .commands.evaluate import evaluate
from .commands.create_split import create_split

# === Setup === #
CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "max_content_width": 120,
    "show_default": True,
}

RESOLUTION_CHOICES = ["Ultra_Low", "Low", "High", "Ultra"]
MODEL_CHOICES = ["Attention_MIL", "TransMIL", "BistroTransformer"]

# === CLI === #
@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version="1.0.0", prog_name="AutoMIL")
def AutoMIL():
    """AutoMIL: Automated Multiple Instance Learning for Whole Slide Images."""
    pass

AutoMIL.add_command(run_pipeline)
AutoMIL.add_command(train)
AutoMIL.add_command(predict)
AutoMIL.add_command(evaluate)
AutoMIL.add_command(create_split)

def main():
    """Entry point for the automil package"""
    AutoMIL.main()

if __name__ == '__main__':
    main()






