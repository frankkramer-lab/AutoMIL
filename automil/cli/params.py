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
# automil/cli/custom_click_params.py

# --- Internal libraries --- #
from typing import Any, Callable, Iterable
import importlib

# --- External libraries --- #
import click
from click import shell_completion

class LazyChoice(click.ParamType):
    """Custom click parameter for lazily-loaded choice options from an import statement.

    This parameter defers importing a module until the parameter is
    evaluated. Choices are derived from a specified attribute or
    callable and cached for subsequent use.

    This is useful for avoiding import overhead in the CLI while
    maintaining validation.

    Example:

    The ``slideflow.norm.StainNormalizer`` class exposes available
    normalizers via an internal dictionary mapping classes to string identifiers:

    ```python

        StainNormalizer.normalizers = {
            "macenko": MacenkoNormalizer,
            "reinhard": ReinhardNormalizer,
            ...
        }
    ```
    
    A ``LazyChoice`` can be configured to expose these keys as CLI options
    without importing Slideflow at startup:

    ```python

        StainNormalizerType = LazyChoice(
            import_path="slideflow.norm",
            attr="StainNormalizer",
            transform=lambda cls: cls.normalizers.keys(),
        )
    ```

    This allows users to select a normalization method via:

    ```bash

        automil run-pipeline ... --stain-normalizer macenko
    ```
    """
    name = "lazy_choice"

    def __init__(
        self,
        import_path: str,
        attribute: str,
        transform: Callable[[Any], Iterable[str]] | None = None
    ):
        self.import_path = import_path
        self.attribute = attribute
        self.transform = transform if transform else (lambda x: x)
        self._choices = None
    
    @property
    def choices(self):
        return self._load_choices()

    def _load_choices(
        self
    ) -> Iterable[str]:
        if self._choices is not None: return self._choices

        module = importlib.import_module(
            self.import_path
        )

        if not (attribute := getattr(
            module,
            self.attribute
        )):
            raise ImportError(f"{module.__name__} does not have a {self.attribute} attribute")
        
        values = self.transform(attribute)
        self._choices = tuple(values)
        return self._choices

    def convert(self, value, param, ctx):
        choices = self._load_choices()

        if value not in choices:
            self.fail(
                f"{value!r} is not a valid choice. "
                f"Choose from: {', '.join(choices)}",
                param,
                ctx,
            )
        return value

    def get_metavar(self, param, ctx):
        return "NORMALIZER"

    def shell_complete(self, ctx, param, incomplete):
        choices = self._load_choices()
        return [
            shell_completion.CompletionItem(c)
            for c in choices if c.startswith(incomplete)
        ]
    
        
