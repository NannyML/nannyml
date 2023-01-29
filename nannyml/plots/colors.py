#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing NannyML-style color options."""
from enum import Enum

import matplotlib


class Colors(str, Enum):
    """Color presets for plotting."""

    INDIGO_PERSIAN = "#3b0280"
    BLUE_SKY_CRAYOLA = "#00c8e5"
    RED_IMPERIAL = "#DD4040"
    SAFFRON = "#E1BC29"
    GREEN_SEA = "#3BB273"
    GRAY_DARK = "#666666"
    GRAY = "#E4E4E4"
    LIGHT_GRAY = "#F5F5F5"

    def transparent(self, alpha: float = 0.2) -> str:
        return 'rgba{}'.format(matplotlib.colors.to_rgba(matplotlib.colors.to_rgb(self.value), alpha))
