from dataclasses import dataclass
from enum import Enum
import math

'''
This module defines standard rebar properties for both metric and imperial systems. 
It includes a RebarSpec dataclass to encapsulate rebar specifications, and dictionaries to map rebar specs to their properties. 
The data in this module is drawn from standard engineering references, including ASTM, ACI 318-19 and CRSI manuals.  
However, there are small discrepancies between sources — especially the rough heuristic associating US bar sizes with diameters measured in eighths of an inch. 
As such, users should verify values against their specific requirements.

A sample use of this function would appear as follows:

from edca_code.constants.rebar import lookup_rebar, UnitSystem

bar = lookup_rebar("H6@300", UnitSystem.METRIC)
print(bar.area_single_bar_mm2)            # mm^2 of one bar
print(bar.area_per_m_width_m2())          # m^2 of steel per meter of width
print(bar.area_over_width_m2(2.5))        # area in m^2 for 2.5 m width

This function will be primarily adopted in the edca_code.scripts.code_checks module for reinforcement calculations during EC or ACI code checks. 
'''

class UnitSystem(str, Enum):
    METRIC = "metric"
    IMPERIAL = "imperial"

@dataclass(frozen=True)
class RebarSpec:
    spec: str                 # e.g. "H6@300" or "#4@12in"
    bar_type: str             # e.g. "H" or "deformed" or "#"
    diameter: float           # numeric diameter (value)
    diameter_unit: str        # "mm" or "in"
    spacing: float            # spacing numeric
    spacing_unit: str         # "m" or "in" (or "mm")
    meta: dict = None

    @property
    def diameter_mm(self) -> float:
        if self.diameter_unit == "mm":
            return self.diameter
        if self.diameter_unit == "in":
            return self.diameter * 25.4
        raise ValueError("unknown diameter unit")

    @property
    def diameter_in(self) -> float:
        if self.diameter_unit == "in":
            return self.diameter
        if self.diameter_unit == "mm":
            return self.diameter / 25.4
        raise ValueError("unknown diameter unit")

    @property
    def area_mm2(self) -> float:
        # area for bars (mm^2)
        return (self.diameter_mm / 2.0) ** 2 * math.pi

    @property
    def area_in2(self) -> float:
        # area in in^2
        return (self.diameter_in / 2.0) ** 2 * math.pi
    
    @property
    def bar_number_per_m(self) -> float:
        if self.spacing_unit == "per_m":
            return 1 / self.spacing
        if self.spacing_unit == "per_mm":
            return 1 / (self.spacing / 1000.0)
        if self.spacing_unit == "per_ft":
            return 1 / (self.spacing * 0.3048)
        raise ValueError("unknown spacing unit")

    @property
    def bar_number_per_ft(self) -> float:
        if self.spacing_unit == "per_ft":
            return 1 / self.spacing
        if self.spacing_unit == "per_m":
            return 1 / (self.spacing * 0.3048)
        if self.spacing_unit == "per_mm":
            return 1 / (self.spacing / 1000.0 * 0.3048)
        raise ValueError("unknown spacing unit")

    def as_dict(self):
        return {
            "spec": self.spec,
            "bar_type": self.bar_type,
            "diameter": self.diameter,
            "diameter_unit": self.diameter_unit,
            "spacing": self.spacing,
            "spacing_unit": self.spacing_unit,
            "area_mm2": self.area_mm2,
            "area_in2": self.area_in2,
            "bar_number": self.bar_number,
            "bar_number_unit": self.bar_number_unit,
            "meta": self.meta or {},
        }

# Below is a partial list of metric rebar specifications. Additional rebar can be added as needed. 
# The first entry is broken down into separate rows for clarity.
METRIC_REBAR_BY_SPEC = {
    # Sample row (H6 at 300 mm spacing)
    "H6@300": RebarSpec(
        spec="H6@300",
        bar_type="H",
        diameter=6.0,
        diameter_unit="mm",
        spacing=300.0,
        spacing_unit="mm",
    ),
    "H6@250": RebarSpec(spec="H6@250", bar_type="H", diameter=6.0, diameter_unit="mm", spacing=250.0, spacing_unit="mm", bar_number=4.0, bar_number_unit="per_m"),
    "H6@200": RebarSpec(spec="H6@200", bar_type="H", diameter=6.0, diameter_unit="mm", spacing=200.0, spacing_unit="mm", bar_number=5.0, bar_number_unit="per_m"),
    "H8@300": RebarSpec(spec="H8@300", bar_type="H", diameter=8.0, diameter_unit="mm", spacing=300.0, spacing_unit="mm", bar_number=3.33, bar_number_unit="per_m"),
    "H6@175": RebarSpec(spec="H6@175", bar_type="H", diameter=6.0, diameter_unit="mm", spacing=175.0, spacing_unit="mm", bar_number=5.71, bar_number_unit="per_m"),
    "H6@150": RebarSpec(spec="H6@150", bar_type="H", diameter=6.0, diameter_unit="mm", spacing=150.0, spacing_unit="mm", bar_number=6.67, bar_number_unit="per_m"),
    "H8@250": RebarSpec(spec="H8@250", bar_type="H", diameter=8.0, diameter_unit="mm", spacing=250.0, spacing_unit="mm", bar_number=4.0, bar_number_unit="per_m"),
    "H6@125": RebarSpec(spec="H6@125", bar_type="H", diameter=6.0, diameter_unit="mm", spacing=125.0, spacing_unit="mm", bar_number=8.0, bar_number_unit="per_m"),
    "H10@300": RebarSpec(spec="H10@300", bar_type="H", diameter=10.0, diameter_unit="mm", spacing=300.0, spacing_unit="mm", bar_number=3.33, bar_number_unit="per_m"),
    "H8@200": RebarSpec(spec="H8@200", bar_type="H", diameter=8.0, diameter_unit="mm", spacing=200.0, spacing_unit="mm", bar_number=5.0, bar_number_unit="per_m"),
    "H6@100": RebarSpec(spec="H6@100", bar_type="H", diameter=6.0, diameter_unit="mm", spacing=100.0, spacing_unit="mm", bar_number=10.0, bar_number_unit="per_m"),
    "H8@175": RebarSpec(spec="H8@175", bar_type="H", diameter=8.0, diameter_unit="mm", spacing=175.0, spacing_unit="mm", bar_number=5.71, bar_number_unit="per_m"),
    "H10@250": RebarSpec(spec="H10@250", bar_type="H", diameter=10.0, diameter_unit="mm", spacing=250.0, spacing_unit="mm", bar_number=4.0, bar_number_unit="per_m"),
    "H12@300": RebarSpec(spec="H12@300", bar_type="H", diameter=12.0, diameter_unit="mm", spacing=300.0, spacing_unit="mm", bar_number=3.33, bar_number_unit="per_m"),
    "H8@150": RebarSpec(spec="H8@150", bar_type="H", diameter=8.0, diameter_unit="mm", spacing=150.0, spacing_unit="mm", bar_number=6.67, bar_number_unit="per_m"),
    "H6@75": RebarSpec(spec="H6@75", bar_type="H", diameter=6.0, diameter_unit="mm", spacing=75.0, spacing_unit="mm", bar_number=13.33, bar_number_unit="per_m"),
    "H10@200": RebarSpec(spec="H10@200", bar_type="H", diameter=10.0, diameter_unit="mm", spacing=200.0, spacing_unit="mm", bar_number=5.0, bar_number_unit="per_m"),
    "H8@125": RebarSpec(spec="H8@125", bar_type="H", diameter=8.0, diameter_unit="mm", spacing=125.0, spacing_unit="mm", bar_number=8.0, bar_number_unit="per_m"),
    "H12@250": RebarSpec(spec="H12@250", bar_type="H", diameter=12.0, diameter_unit="mm", spacing=250.0, spacing_unit="mm", bar_number=4.0, bar_number_unit="per_m"),
    "H14@300": RebarSpec(spec="H14@300", bar_type="H", diameter=14.0, diameter_unit="mm", spacing=300.0, spacing_unit="mm", bar_number=3.33, bar_number_unit="per_m"),
    "H10@175": RebarSpec(spec="H10@175", bar_type="H", diameter=10.0, diameter_unit="mm", spacing=175.0, spacing_unit="mm", bar_number=5.71, bar_number_unit="per_m"),
    "H8@100": RebarSpec(spec="H8@100", bar_type="H", diameter=8.0, diameter_unit="mm", spacing=100.0, spacing_unit="mm", bar_number=10.0, bar_number_unit="per_m"),
    "H10@150": RebarSpec(spec="H10@150", bar_type="H", diameter=10.0, diameter_unit="mm", spacing=150.0, spacing_unit="mm", bar_number=6.67, bar_number_unit="per_m"),
    "H12@200": RebarSpec(spec="H12@200", bar_type="H", diameter=12.0, diameter_unit="mm", spacing=200.0, spacing_unit="mm", bar_number=5.0, bar_number_unit="per_m"),
    "H16@300": RebarSpec(spec="H16@300", bar_type="H", diameter=16.0, diameter_unit="mm", spacing=300.0, spacing_unit="mm", bar_number=3.33, bar_number_unit="per_m"),
    "H14@250": RebarSpec(spec="H14@250", bar_type="H", diameter=14.0, diameter_unit="mm", spacing=250.0, spacing_unit="mm", bar_number=4.0, bar_number_unit="per_m"),
    "H10@125": RebarSpec(spec="H10@125", bar_type="H", diameter=10.0, diameter_unit="mm", spacing=125.0, spacing_unit="mm", bar_number=8.0, bar_number_unit="per_m"),
    "H8@75": RebarSpec(spec="H8@75", bar_type="H", diameter=8.0, diameter_unit="mm", spacing=75.0, spacing_unit="mm", bar_number=13.33, bar_number_unit="per_m"),
    "H12@175": RebarSpec(spec="H12@175", bar_type="H", diameter=12.0, diameter_unit="mm", spacing=175.0, spacing_unit="mm", bar_number=5.71, bar_number_unit="per_m"),
    "H18@300": RebarSpec(spec="H18@300", bar_type="H", diameter=18.0, diameter_unit="mm", spacing=300.0, spacing_unit="mm", bar_number=3.33, bar_number_unit="per_m"),
    "H14@200": RebarSpec(spec="H14@200", bar_type="H", diameter=14.0, diameter_unit="mm", spacing=200.0, spacing_unit="mm", bar_number=5.0, bar_number_unit="per_m"),
    "H10@100": RebarSpec(spec="H10@100", bar_type="H", diameter=10.0, diameter_unit="mm", spacing=100.0, spacing_unit="mm", bar_number=10.0, bar_number_unit="per_m"),
    "H12@150": RebarSpec(spec="H12@150", bar_type="H", diameter=12.0, diameter_unit="mm", spacing=150.0, spacing_unit="mm", bar_number=6.67, bar_number_unit="per_m"),
    "H16@250": RebarSpec(spec="H16@250", bar_type="H", diameter=16.0, diameter_unit="mm", spacing=250.0, spacing_unit="mm", bar_number=4.0, bar_number_unit="per_m"),
    "H12@125": RebarSpec(spec="H12@125", bar_type="H", diameter=12.0, diameter_unit="mm", spacing=125.0, spacing_unit="mm", bar_number=8.0, bar_number_unit="per_m"),
    "H14@175": RebarSpec(spec="H14@175", bar_type="H", diameter=14.0, diameter_unit="mm", spacing=175.0, spacing_unit="mm", bar_number=5.71, bar_number_unit="per_m"),
    "H20@300": RebarSpec(spec="H20@300", bar_type="H", diameter=20.0, diameter_unit="mm", spacing=300.0, spacing_unit="mm", bar_number=3.33, bar_number_unit="per_m"),
    "H16@200": RebarSpec(spec="H16@200", bar_type="H", diameter=16.0, diameter_unit="mm", spacing=200.0, spacing_unit="mm", bar_number=5.0, bar_number_unit="per_m"),
    "H18@250": RebarSpec(spec="H18@250", bar_type="H", diameter=18.0, diameter_unit="mm", spacing=250.0, spacing_unit="mm", bar_number=4.0, bar_number_unit="per_m"),
    "H10@75": RebarSpec(spec="H10@75", bar_type="H", diameter=10.0, diameter_unit="mm", spacing=75.0, spacing_unit="mm", bar_number=13.33, bar_number_unit="per_m"),
    "H14@150": RebarSpec(spec="H14@150", bar_type="H", diameter=14.0, diameter_unit="mm", spacing=150.0, spacing_unit="mm", bar_number=6.67, bar_number_unit="per_m"),
    "H12@100": RebarSpec(spec="H12@100", bar_type="H", diameter=12.0, diameter_unit="mm", spacing=100.0, spacing_unit="mm", bar_number=10.0, bar_number_unit="per_m"),
    "H22@300": RebarSpec(spec="H22@300", bar_type="H", diameter=22.0, diameter_unit="mm", spacing=300.0, spacing_unit="mm", bar_number=3.33, bar_number_unit="per_m"),
    "H16@175": RebarSpec(spec="H16@175", bar_type="H", diameter=16.0, diameter_unit="mm", spacing=175.0, spacing_unit="mm", bar_number=5.71, bar_number_unit="per_m"),
    "H14@125": RebarSpec(spec="H14@125", bar_type="H", diameter=14.0, diameter_unit="mm", spacing=125.0, spacing_unit="mm", bar_number=8.0, bar_number_unit="per_m"),
    "H20@250": RebarSpec(spec="H20@250", bar_type="H", diameter=20.0, diameter_unit="mm", spacing=250.0, spacing_unit="mm", bar_number=4.0, bar_number_unit="per_m"),
    "H18@200": RebarSpec(spec="H18@200", bar_type="H", diameter=18.0, diameter_unit="mm", spacing=200.0, spacing_unit="mm", bar_number=5.0, bar_number_unit="per_m"),
    "H16@150": RebarSpec(spec="H16@150", bar_type="H", diameter=16.0, diameter_unit="mm", spacing=150.0, spacing_unit="mm", bar_number=6.67, bar_number_unit="per_m"),
    "H12@75": RebarSpec(spec="H12@75", bar_type="H", diameter=12.0, diameter_unit="mm", spacing=75.0, spacing_unit="mm", bar_number=13.33, bar_number_unit="per_m"),
    "H25@300": RebarSpec(spec="H25@300", bar_type="H", diameter=25.0, diameter_unit="mm", spacing=300.0, spacing_unit="mm", bar_number=3.33, bar_number_unit="per_m"),
    "H22@250": RebarSpec(spec="H22@250", bar_type="H", diameter=22.0, diameter_unit="mm", spacing=250.0, spacing_unit="mm", bar_number=4.0, bar_number_unit="per_m"),
    "H18@175": RebarSpec(spec="H18@175", bar_type="H", diameter=18.0, diameter_unit="mm", spacing=175.0, spacing_unit="mm", bar_number=5.71, bar_number_unit="per_m"),
    "H14@100": RebarSpec(spec="H14@100", bar_type="H", diameter=14.0, diameter_unit="mm", spacing=100.0, spacing_unit="mm", bar_number=10.0, bar_number_unit="per_m"),
    "H20@200": RebarSpec(spec="H20@200", bar_type="H", diameter=20.0, diameter_unit="mm", spacing=200.0, spacing_unit="mm", bar_number=5.0, bar_number_unit="per_m"),
    "H16@125": RebarSpec(spec="H16@125", bar_type="H", diameter=16.0, diameter_unit="mm", spacing=125.0, spacing_unit="mm", bar_number=8.0, bar_number_unit="per_m"),
    "H18@150": RebarSpec(spec="H18@150", bar_type="H", diameter=18.0, diameter_unit="mm", spacing=150.0, spacing_unit="mm", bar_number=6.67, bar_number_unit="per_m"),
    "H28@300": RebarSpec(spec="H28@300", bar_type="H", diameter=28.0, diameter_unit="mm", spacing=300.0, spacing_unit="mm", bar_number=3.33, bar_number_unit="per_m"),
    "H20@175": RebarSpec(spec="H20@175", bar_type="H", diameter=20.0, diameter_unit="mm", spacing=175.0, spacing_unit="mm", bar_number=5.71, bar_number_unit="per_m"),
    "H22@200": RebarSpec(spec="H22@200", bar_type="H", diameter=22.0, diameter_unit="mm", spacing=200.0, spacing_unit="mm", bar_number=5.0, bar_number_unit="per_m"),
    "H25@250": RebarSpec(spec="H25@250", bar_type="H", diameter=25.0, diameter_unit="mm", spacing=250.0, spacing_unit="mm", bar_number=4.0, bar_number_unit="per_m"),
    "H14@75": RebarSpec(spec="H14@75", bar_type="H", diameter=14.0, diameter_unit="mm", spacing=75.0, spacing_unit="mm", bar_number=13.33, bar_number_unit="per_m"),
    "H16@100": RebarSpec(spec="H16@100", bar_type="H", diameter=16.0, diameter_unit="mm", spacing=100.0, spacing_unit="mm", bar_number=10.0, bar_number_unit="per_m"),
    "H18@125": RebarSpec(spec="H18@125", bar_type="H", diameter=18.0, diameter_unit="mm", spacing=125.0, spacing_unit="mm", bar_number=8.0, bar_number_unit="per_m"),
    "H20@150": RebarSpec(spec="H20@150", bar_type="H", diameter=20.0, diameter_unit="mm", spacing=150.0, spacing_unit="mm", bar_number=6.67, bar_number_unit="per_m"),
    "H22@175": RebarSpec(spec="H22@175", bar_type="H", diameter=22.0, diameter_unit="mm", spacing=175.0, spacing_unit="mm", bar_number=5.71, bar_number_unit="per_m"),
    "H32@300": RebarSpec(spec="H32@300", bar_type="H", diameter=32.0, diameter_unit="mm", spacing=300.0, spacing_unit="mm", bar_number=3.33, bar_number_unit="per_m"),
    "H25@200": RebarSpec(spec="H25@200", bar_type="H", diameter=25.0, diameter_unit="mm", spacing=200.0, spacing_unit="mm", bar_number=5.0, bar_number_unit="per_m"),
    "H28@250": RebarSpec(spec="H28@250", bar_type="H", diameter=28.0, diameter_unit="mm", spacing=250.0, spacing_unit="mm", bar_number=4.0, bar_number_unit="per_m"),
    "H20@125": RebarSpec(spec="H20@125", bar_type="H", diameter=20.0, diameter_unit="mm", spacing=125.0, spacing_unit="mm", bar_number=8.0, bar_number_unit="per_m"),
    "H18@100": RebarSpec(spec="H18@100", bar_type="H", diameter=18.0, diameter_unit="mm", spacing=100.0, spacing_unit="mm", bar_number=10.0, bar_number_unit="per_m"),
    "H16@75": RebarSpec(spec="H16@75", bar_type="H", diameter=16.0, diameter_unit="mm", spacing=75.0, spacing_unit="mm", bar_number=13.33, bar_number_unit="per_m"),
    "H22@150": RebarSpec(spec="H22@150", bar_type="H", diameter=22.0, diameter_unit="mm", spacing=150.0, spacing_unit="mm", bar_number=6.67, bar_number_unit="per_m"),
    "H25@175": RebarSpec(spec="H25@175", bar_type="H", diameter=25.0, diameter_unit="mm", spacing=175.0, spacing_unit="mm", bar_number=5.71, bar_number_unit="per_m"),
    "H22@125": RebarSpec(spec="H22@125", bar_type="H", diameter=22.0, diameter_unit="mm", spacing=125.0, spacing_unit="mm", bar_number=8.0, bar_number_unit="per_m"),
    "H28@200": RebarSpec(spec="H28@200", bar_type="H", diameter=28.0, diameter_unit="mm", spacing=200.0, spacing_unit="mm", bar_number=5.0, bar_number_unit="per_m"),
    "H20@100": RebarSpec(spec="H20@100", bar_type="H", diameter=20.0, diameter_unit="mm", spacing=100.0, spacing_unit="mm", bar_number=10.0, bar_number_unit="per_m"),
    "H32@250": RebarSpec(spec="H32@250", bar_type="H", diameter=32.0, diameter_unit="mm", spacing=250.0, spacing_unit="mm", bar_number=4.0, bar_number_unit="per_m"),
    "H18@75": RebarSpec(spec="H18@75", bar_type="H", diameter=18.0, diameter_unit="mm", spacing=75.0, spacing_unit="mm", bar_number=13.33, bar_number_unit="per_m"),
    "H25@150": RebarSpec(spec="H25@150", bar_type="H", diameter=25.0, diameter_unit="mm", spacing=150.0, spacing_unit="mm", bar_number=6.67, bar_number_unit="per_m"),
    "H28@175": RebarSpec(spec="H28@175", bar_type="H", diameter=28.0, diameter_unit="mm", spacing=175.0, spacing_unit="mm", bar_number=5.71, bar_number_unit="per_m"),
    "H40@300": RebarSpec(spec="H40@300", bar_type="H", diameter=40.0, diameter_unit="mm", spacing=300.0, spacing_unit="mm", bar_number=3.33, bar_number_unit="per_m"),
    "H22@100": RebarSpec(spec="H22@100", bar_type="H", diameter=22.0, diameter_unit="mm", spacing=100.0, spacing_unit="mm", bar_number=10.0, bar_number_unit="per_m"),
    "H25@125": RebarSpec(spec="H25@125", bar_type="H", diameter=25.0, diameter_unit="mm", spacing=125.0, spacing_unit="mm", bar_number=8.0, bar_number_unit="per_m"),
    "H32@200": RebarSpec(spec="H32@200", bar_type="H", diameter=32.0, diameter_unit="mm", spacing=200.0, spacing_unit="mm", bar_number=5.0, bar_number_unit="per_m"),
    "H20@75": RebarSpec(spec="H20@75", bar_type="H", diameter=20.0, diameter_unit="mm", spacing=75.0, spacing_unit="mm", bar_number=13.33, bar_number_unit="per_m"),
    "H28@150": RebarSpec(spec="H28@150", bar_type="H", diameter=28.0, diameter_unit="mm", spacing=150.0, spacing_unit="mm", bar_number=6.67, bar_number_unit="per_m"),
    "H32@175": RebarSpec(spec="H32@175", bar_type="H", diameter=32.0, diameter_unit="mm", spacing=175.0, spacing_unit="mm", bar_number=5.71, bar_number_unit="per_m"),
    "H25@100": RebarSpec(spec="H25@100", bar_type="H", diameter=25.0, diameter_unit="mm", spacing=100.0, spacing_unit="mm", bar_number=10.0, bar_number_unit="per_m"),
    "H28@125": RebarSpec(spec="H28@125", bar_type="H", diameter=28.0, diameter_unit="mm", spacing=125.0, spacing_unit="mm", bar_number=8.0, bar_number_unit="per_m"),
    "H22@75": RebarSpec(spec="H22@75", bar_type="H", diameter=22.0, diameter_unit="mm", spacing=75.0, spacing_unit="mm", bar_number=13.33, bar_number_unit="per_m"),
    "H40@250": RebarSpec(spec="H40@250", bar_type="H", diameter=40.0, diameter_unit="mm", spacing=250.0, spacing_unit="mm", bar_number=4.0, bar_number_unit="per_m"),
    "H32@150": RebarSpec(spec="H32@150", bar_type="H", diameter=32.0, diameter_unit="mm", spacing=150.0, spacing_unit="mm", bar_number=6.67, bar_number_unit="per_m"),
    "H28@100": RebarSpec(spec="H28@100", bar_type="H", diameter=28.0, diameter_unit="mm", spacing=100.0, spacing_unit="mm", bar_number=10.0, bar_number_unit="per_m"),
    "H40@200": RebarSpec(spec="H40@200", bar_type="H", diameter=40.0, diameter_unit="mm", spacing=200.0, spacing_unit="mm", bar_number=5.0, bar_number_unit="per_m"),
    "H25@75": RebarSpec(spec="H25@75", bar_type="H", diameter=25.0, diameter_unit="mm", spacing=75.0, spacing_unit="mm", bar_number=13.33, bar_number_unit="per_m"),
    "H32@125": RebarSpec(spec="H32@125", bar_type="H", diameter=32.0, diameter_unit="mm", spacing=125.0, spacing_unit="mm", bar_number=8.0, bar_number_unit="per_m"),
    "H40@175": RebarSpec(spec="H40@175", bar_type="H", diameter=40.0, diameter_unit="mm", spacing=175.0, spacing_unit="mm", bar_number=5.71, bar_number_unit="per_m"),
    "H28@75": RebarSpec(spec="H28@75", bar_type="H", diameter=28.0, diameter_unit="mm", spacing=75.0, spacing_unit="mm", bar_number=13.33, bar_number_unit="per_m"),
    "H32@100": RebarSpec(spec="H32@100", bar_type="H", diameter=32.0, diameter_unit="mm", spacing=100.0, spacing_unit="mm", bar_number=10.0, bar_number_unit="per_m"),
    "H40@150": RebarSpec(spec="H40@150", bar_type="H", diameter=40.0, diameter_unit="mm", spacing=150.0, spacing_unit="mm", bar_number=6.67, bar_number_unit="per_m"),
    "H40@125": RebarSpec(spec="H40@125", bar_type="H", diameter=40.0, diameter_unit="mm", spacing=125.0, spacing_unit="mm", bar_number=8.0, bar_number_unit="per_m"),
    "H32@75": RebarSpec(spec="H32@75", bar_type="H", diameter=32.0, diameter_unit="mm", spacing=75.0, spacing_unit="mm", bar_number=13.33, bar_number_unit="per_m"),
    "H40@100": RebarSpec(spec="H40@100", bar_type="H", diameter=40.0, diameter_unit="mm", spacing=100.0, spacing_unit="mm", bar_number=10.0, bar_number_unit="per_m"),
    "H40@75": RebarSpec(spec="H40@75", bar_type="H", diameter=40.0, diameter_unit="mm", spacing=75.0, spacing_unit="mm", bar_number=13.33, bar_number_unit="per_m"),
}

# Build index by normalized diameter (mm)
METRIC_BY_DIAMETER_MM = {}
for diameter in METRIC_REBAR_BY_SPEC.values():
    METRIC_BY_DIAMETER_MM.setdefault(diameter.diameter, []).append(diameter)

IMPERIAL_REBAR_BY_SPEC = {
    # --- #3 ---
    "#3@6in":  RebarSpec("#3@6in",  "#3", 0.375, "in", 6.0,  "in"),
    "#3@8in":  RebarSpec("#3@8in",  "#3", 0.375, "in", 8.0,  "in"),
    "#3@12in": RebarSpec("#3@12in", "#3", 0.375, "in", 12.0, "in"),
    "#3@16in": RebarSpec("#3@16in", "#3", 0.375, "in", 16.0, "in"),

    # --- #4 ---
    "#4@6in":  RebarSpec("#4@6in",  "#4", 0.500, "in", 6.0,  "in"),
    "#4@8in":  RebarSpec("#4@8in",  "#4", 0.500, "in", 8.0,  "in"),
    "#4@12in": RebarSpec("#4@12in", "#4", 0.500, "in", 12.0, "in"),
    "#4@16in": RebarSpec("#4@16in", "#4", 0.500, "in", 16.0, "in"),

    # --- #5 ---
    "#5@6in":  RebarSpec("#5@6in",  "#5", 0.625, "in", 6.0,  "in"),
    "#5@8in":  RebarSpec("#5@8in",  "#5", 0.625, "in", 8.0,  "in"),
    "#5@12in": RebarSpec("#5@12in", "#5", 0.625, "in", 12.0, "in"),
    "#5@16in": RebarSpec("#5@16in", "#5", 0.625, "in", 16.0, "in"),

    # --- #6 ---
    "#6@6in":  RebarSpec("#6@6in",  "#6", 0.750, "in", 6.0,  "in"),
    "#6@8in":  RebarSpec("#6@8in",  "#6", 0.750, "in", 8.0,  "in"),
    "#6@12in": RebarSpec("#6@12in", "#6", 0.750, "in", 12.0, "in"),
    "#6@16in": RebarSpec("#6@16in", "#6", 0.750, "in", 16.0, "in"),

    # --- #7 ---
    "#7@6in":  RebarSpec("#7@6in",  "#7", 0.875, "in", 6.0,  "in"),
    "#7@8in":  RebarSpec("#7@8in",  "#7", 0.875, "in", 8.0,  "in"),
    "#7@12in": RebarSpec("#7@12in", "#7", 0.875, "in", 12.0, "in"),
    "#7@16in": RebarSpec("#7@16in", "#7", 0.875, "in", 16.0, "in"),

    # --- #8 ---
    "#8@6in":  RebarSpec("#8@6in",  "#8", 1.000, "in", 6.0,  "in"),
    "#8@8in":  RebarSpec("#8@8in",  "#8", 1.000, "in", 8.0,  "in"),
    "#8@12in": RebarSpec("#8@12in", "#8", 1.000, "in", 12.0, "in"),
    "#8@16in": RebarSpec("#8@16in", "#8", 1.000, "in", 16.0, "in"),

    # --- #9 ---
    "#9@6in":  RebarSpec("#9@6in",  "#9", 1.128, "in", 6.0,  "in"),
    "#9@8in":  RebarSpec("#9@8in",  "#9", 1.128, "in", 8.0,  "in"),
    "#9@12in": RebarSpec("#9@12in", "#9", 1.128, "in", 12.0, "in"),
    "#9@16in": RebarSpec("#9@16in", "#9", 1.128, "in", 16.0, "in"),

    # --- #10 ---
    "#10@6in":  RebarSpec("#10@6in",  "#10", 1.270, "in", 6.0,  "in"),
    "#10@8in":  RebarSpec("#10@8in",  "#10", 1.270, "in", 8.0,  "in"),
    "#10@12in": RebarSpec("#10@12in", "#10", 1.270, "in", 12.0, "in"),
    "#10@16in": RebarSpec("#10@16in", "#10", 1.270, "in", 16.0, "in"),

    # --- #11 ---
    "#11@6in":  RebarSpec("#11@6in",  "#11", 1.410, "in", 6.0,  "in"),
    "#11@8in":  RebarSpec("#11@8in",  "#11", 1.410, "in", 8.0,  "in"),
    "#11@12in": RebarSpec("#11@12in", "#11", 1.410, "in", 12.0, "in"),
    "#11@16in": RebarSpec("#11@16in", "#11", 1.410, "in", 16.0, "in"),

    # --- #14 ---
    "#14@6in":  RebarSpec("#14@6in",  "#14", 1.693, "in", 6.0,  "in"),
    "#14@8in":  RebarSpec("#14@8in",  "#14", 1.693, "in", 8.0,  "in"),
    "#14@12in": RebarSpec("#14@12in", "#14", 1.693, "in", 12.0, "in"),
    "#14@16in": RebarSpec("#14@16in", "#14", 1.693, "in", 16.0, "in"),

    # --- #18 ---
    "#18@6in":  RebarSpec("#18@6in",  "#18", 2.257, "in", 6.0,  "in"),
    "#18@8in":  RebarSpec("#18@8in",  "#18", 2.257, "in", 8.0,  "in"),
    "#18@12in": RebarSpec("#18@12in", "#18", 2.257, "in", 12.0, "in"),
    "#18@16in": RebarSpec("#18@16in", "#18", 2.257, "in", 16.0, "in"),
}

# Build index by normalized diameter (in)
IMPERIAL_BY_DIAMETER_IN = {}
for diameter in IMPERIAL_REBAR_BY_SPEC.values():
    IMPERIAL_BY_DIAMETER_IN.setdefault(diameter.diameter, []).append(diameter)