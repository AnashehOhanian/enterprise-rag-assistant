"""Material catalog and early price brackets for structural steel, gates, stairs, patios, and railings.

The tables are the stock list used for takeoffs: rolled profiles, hollow
sections and pipe, plate thicknesses, gate styles, stair styles, patio
structures, railing styles, and nominal weight. W, S, and C weights are the nominal pounds per foot in the designation.
Angle and HSS weights are nominal catalog values. Pipe weights follow
ASME B36.10. Plate weight uses 40.8 pounds per square foot for each inch
of thickness, the shop rounding of 490 lb/ft³.

``estimate_price`` returns a low-to-high bracket and has no single price.
Material estimates apply a planning band per pound, then a cut-and-market
allowance on the high edge only. Gate estimates apply a band per square
foot of clear opening. Stair estimates add up a planning weight for the
flight, then apply that style's shop-fabrication band per pound. Patio estimates name catalog
profiles for the footprint, add up that steel, and apply the structure's
shop-fabrication band per pound. Railing estimates name posts, rails,
infill, and mounting plates for a run, then apply that style's
shop-fabrication band per pound. Both edges are rounded outward so the
bracket contains the raw calculation. The result is an early estimate, not a quote.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

STEEL_DENSITY_LB_PER_CU_FT = 490.0
STEEL_LB_PER_SQ_FT_PER_INCH = 40.8

# Planning band for steel supplied cut-to-length, USD per pound.
# The high edge is widened again by HIGH_SIDE_ALLOWANCE.
MATERIAL_LOW_USD_PER_LB = 0.75
MATERIAL_HIGH_USD_PER_LB = 1.35
HIGH_SIDE_ALLOWANCE = 0.10

# Planning geometry for a stair when the takeoff does not override it.
# Rise and run are the usual 7-11 pair. A spiral uses the rise only and
# plans each tread as a 30 degree sector around the center column.
STAIR_DEFAULT_RISE_IN = 7.0
STAIR_DEFAULT_RUN_IN = 11.0
STAIR_SPIRAL_DEGREES_PER_STEP = 30.0
STAIR_SPIRAL_INNER_RADIUS_IN = 4.0

# Planning layout for an outdoor frame. Span is the rafter direction.
# The other side is area divided by span. Posts sit about 10 ft apart.
PATIO_POST_SPACING_FT = 10.0
PATIO_POST_HEIGHT_FT = 8.0
PATIO_RAILING_POST_SPACING_FT = 6.0
PATIO_RAILING_HEIGHT_FT = 3.5

_MAX_WEIGHT_LB = 1_000_000.0
_MAX_LENGTH_FT = 1_000.0
_MAX_PLATE_SIDE_IN = 480.0
_MAX_THICKNESS_IN = 8.0
_MAX_OPENING_FT = 120.0
_PLATE_MATCH_TOLERANCE_IN = 0.001
_MIN_STEPS = 2
_MAX_STEPS = 36
_MIN_STAIR_WIDTH_IN = 24.0
_MAX_STAIR_WIDTH_IN = 120.0
_MIN_SPIRAL_DIAMETER_IN = 48.0
_MAX_SPIRAL_DIAMETER_IN = 96.0
_MIN_RISE_IN = 5.0
_MAX_RISE_IN = 9.0
_MIN_RUN_IN = 9.0
_MAX_RUN_IN = 14.0
_MIN_PATIO_SPAN_FT = 8.0
_MAX_PATIO_SPAN_FT = 24.0
_MIN_PATIO_AREA_SQ_FT = 64.0
_MAX_PATIO_AREA_SQ_FT = 1500.0
_MIN_PATIO_WIDTH_FT = 8.0
_MAX_PATIO_WIDTH_FT = 60.0

# A railing run is priced from linear footage and height. Posts include both
# ends. Pickets are 1/2 in square bar at 4.5 in centers. Cables are 1/8 in
# runs at 3 in, with a fitting at each end. Mounts are catalog plate.
RAILING_PICKET_SPACING_IN = 4.5
RAILING_HORIZONTAL_SPACING_IN = 4.0
RAILING_ORNAMENTAL_SPACING_IN = 4.0
RAILING_PICKET_LB_PER_FT = 0.85
RAILING_CABLE_SPACING_IN = 3.0
RAILING_CABLE_LB_PER_FT = 0.04
RAILING_FITTING_LB = 0.25
RAILING_DEFAULT_MOUNT = "base_plate"
_MIN_RAILING_LENGTH_FT = 4.0
_MAX_RAILING_LENGTH_FT = 300.0
_MIN_RAILING_HEIGHT_FT = 2.5
_MAX_RAILING_HEIGHT_FT = 5.0

_FRACTION_NAMES = {
    0.125: "1/8",
    0.1875: "3/16",
    0.25: "1/4",
    0.3125: "5/16",
    0.375: "3/8",
    0.5: "1/2",
    0.625: "5/8",
    0.75: "3/4",
}


class CatalogError(ValueError):
    """The catalog cannot build a price bracket for this request."""


@dataclass(frozen=True)
class SteelProfile:
    """A rolled section sold by the foot."""

    designation: str
    family: str
    weight_lb_per_ft: float


@dataclass(frozen=True)
class TubeSize:
    """Square HSS, rectangular HSS, or round pipe sold by the foot."""

    designation: str
    shape: str
    weight_lb_per_ft: float
    outside_width_in: float
    outside_height_in: float
    wall_in: float
    schedule: str | None = None
    nominal_size: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlateThickness:
    """A plate or sheet thickness and its weight per square foot."""

    name: str
    thickness_in: float
    weight_lb_per_sq_ft: float
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateStyle:
    """A gate type and its shop-fabrication planning band.

    Rates are USD per square foot of clear opening. Minimums apply per leaf
    assembly before quantity is applied. The band covers a primed frame,
    infill, and standard hinges or rollers. It does not cover operators,
    footings, freight, permits, or galvanizing.
    """

    key: str
    name: str
    description: str
    leaf_count: int
    low_usd_per_sq_ft: float
    high_usd_per_sq_ft: float
    minimum_low_usd: float
    minimum_high_usd: float
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class StairStyle:
    """A custom stair type and the shop weight used to price it.

    Stringer weight is pounds per foot of slope, or per foot of column
    height on a spiral, plus an extra amount for each inch of width.
    Tread and riser weights are pounds per square foot of plate or pan.
    Rates are USD per pound of that calculated steel, for one shop-primed
    flight. Minimums apply per flight before quantity. The band covers
    stringers, treads, risers when the style has them, and connectors.
    It does not cover handrail, guard, landings, wall structure, or installation.
    """

    key: str
    name: str
    description: str
    layout: str
    stringer_count: int
    stringer_lb_per_ft: float
    stringer_lb_per_ft_per_in: float
    tread_lb_per_sq_ft: float
    tread_depth_in: float
    riser_lb_per_sq_ft: float
    connector_lb_per_step: float
    low_usd_per_lb: float
    high_usd_per_lb: float
    minimum_low_usd: float
    minimum_high_usd: float
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class StairTakeoff:
    """The steel weight behind one stair bracket.

    Component weights are for a single flight. ``weight_lb`` multiplies
    them by quantity.
    """

    style: StairStyle
    step_count: int
    width_in: float
    rise_in: float
    run_in: float | None
    stringer_lb: float
    tread_lb: float
    riser_lb: float
    connector_lb: float
    quantity: int = 1

    @property
    def each_lb(self) -> float:
        """Return the steel weight of one flight."""
        return self.stringer_lb + self.tread_lb + self.riser_lb + self.connector_lb

    @property
    def weight_lb(self) -> float:
        """Return the steel weight of every flight in the takeoff."""
        return self.each_lb * self.quantity


@dataclass(frozen=True)
class PatioStyle:
    """An outdoor structure and the shop band used to price its frame.

    Rates are USD per pound of the catalog profiles selected for the
    footprint. Minimums apply per structure before quantity. The band covers
    a shop-primed frame. It does not cover roof panels, footings, flashing,
    anchors, or installation.
    """

    key: str
    name: str
    description: str
    kind: str
    low_usd_per_lb: float
    high_usd_per_lb: float
    minimum_low_usd: float
    minimum_high_usd: float
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class PatioLine:
    """One profile group in an outdoor frame."""

    role: str
    designation: str
    count: int
    length_ft: float
    weight_lb: float


@dataclass(frozen=True)
class PatioTakeoff:
    """The named profiles and steel weight behind one patio bracket.

    Line weights are for a single structure. ``weight_lb`` multiplies them
    by quantity.
    """

    style: PatioStyle
    area_sq_ft: float
    span_ft: float
    width_ft: float
    lines: tuple[PatioLine, ...]
    quantity: int = 1

    @property
    def each_lb(self) -> float:
        """Return the steel weight of one structure."""
        return sum(line.weight_lb for line in self.lines)

    @property
    def weight_lb(self) -> float:
        """Return the steel weight of every structure in the takeoff."""
        return self.each_lb * self.quantity


@dataclass(frozen=True)
class RailingMount:
    """Hardware used once at each railing post.

    A plate mount is catalog plate. A member mount is a short catalog angle
    or tube. Anchors, core drilling, and installation stay off the bracket.
    """

    key: str
    name: str
    description: str
    kind: str
    stock: str
    width_in: float
    length_in: float
    length_ft: float
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class RailingStyle:
    """A railing or balustrade and the shop band used to price the run.

    Rates are USD per pound of the posts, rails, infill, and the selected
    mounting hardware. Minimums apply per run before quantity. The band
    covers a shop-primed railing. It does not cover anchors, core drilling,
    glass, a wood cap, coatings beyond primer, or installation.
    """

    key: str
    name: str
    description: str
    kind: str
    post: str
    top_rail: str
    bottom_rail: str
    post_spacing_ft: float
    default_height_ft: float
    infill: str
    low_usd_per_lb: float
    high_usd_per_lb: float
    minimum_low_usd: float
    minimum_high_usd: float
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class RailingLine:
    """One group of steel or hardware in a railing run."""

    role: str
    designation: str
    count: int
    length_ft: float
    weight_lb: float


@dataclass(frozen=True)
class RailingTakeoff:
    """The named steel and mounts behind one railing bracket.

    Line weights are for a single run. ``weight_lb`` multiplies them by quantity.
    """

    style: RailingStyle
    mount: RailingMount
    length_ft: float
    height_ft: float
    lines: tuple[RailingLine, ...]
    quantity: int = 1

    @property
    def each_lb(self) -> float:
        """Return the steel and hardware weight of one run."""
        return sum(line.weight_lb for line in self.lines)

    @property
    def weight_lb(self) -> float:
        """Return the weight of every run in the takeoff."""
        return self.each_lb * self.quantity


@dataclass(frozen=True)
class PriceBracket:
    """A low-to-high estimate. There is no midpoint to quote."""

    low_usd: float
    high_usd: float
    basis: str
    summary: str
    weight_lb: float | None = None
    quantity: int = 1

    @property
    def label(self) -> str:
        """Return the bracket in dollars, such as ``$390.00–$772.20``."""
        return f"${self.low_usd:,.2f}–${self.high_usd:,.2f}"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dictionary without a single price."""
        return {
            "low_usd": self.low_usd,
            "high_usd": self.high_usd,
            "range": self.label,
            "basis": self.basis,
            "summary": self.summary,
            "weight_lb": self.weight_lb,
            "quantity": self.quantity,
        }


def _dim(value: float) -> str:
    """Format a catalog dimension with a common fraction when it has one."""
    whole = int(value)
    fraction = round(value - whole, 4)
    if fraction == 0:
        return str(whole)
    name = _FRACTION_NAMES.get(fraction)
    if name is None:
        raise RuntimeError(f"No fraction name for {value}.")
    if whole == 0:
        return name
    return f"{whole}-{name}"


def _rolled(family: str, designation: str) -> SteelProfile:
    """Build a W, S, or C profile whose weight is the designation number."""
    prefix = {"wide_flange": "W", "american_standard": "S", "channel": "C"}[family]
    if not designation.startswith(prefix) or "x" not in designation:
        raise RuntimeError(f"Unexpected {family} designation {designation!r}.")
    weight = float(designation.split("x")[-1])
    return SteelProfile(designation, family, weight)


def _angle(designation: str, weight_lb_per_ft: float) -> SteelProfile:
    """Build an angle. Its designation does not encode the weight."""
    if not designation.startswith("L"):
        raise RuntimeError(f"Unexpected angle designation {designation!r}.")
    return SteelProfile(designation, "angle", weight_lb_per_ft)


def _hss(width_in: float, height_in: float, wall_in: float, weight_lb_per_ft: float) -> TubeSize:
    """Build a square or rectangular HSS entry."""
    designation = f"HSS{_dim(width_in)}x{_dim(height_in)}x{_dim(wall_in)}"
    shape = "square" if width_in == height_in else "rectangular"
    return TubeSize(
        designation=designation,
        shape=shape,
        weight_lb_per_ft=weight_lb_per_ft,
        outside_width_in=width_in,
        outside_height_in=height_in,
        wall_in=wall_in,
    )


def _pipe(
    nominal_size: str,
    schedule: str,
    outside_in: float,
    wall_in: float,
    weight_lb_per_ft: float,
) -> TubeSize:
    """Build a round pipe entry and the names a takeoff is likely to use."""
    designation = f"PIPE {nominal_size} SCH {schedule}"
    return TubeSize(
        designation=designation,
        shape="pipe",
        weight_lb_per_ft=weight_lb_per_ft,
        outside_width_in=outside_in,
        outside_height_in=outside_in,
        wall_in=wall_in,
        schedule=schedule,
        nominal_size=nominal_size,
        aliases=(
            f"{nominal_size} SCH {schedule}",
            f"NPS {nominal_size} SCH {schedule}",
        ),
    )


def _plate(name: str, thickness_in: float, *aliases: str) -> PlateThickness:
    """Build a plate row. Weight is thickness times the shop factor."""
    weight = round(thickness_in * STEEL_LB_PER_SQ_FT_PER_INCH, 2)
    return PlateThickness(name, thickness_in, weight, aliases)


_WIDE_FLANGE = (
    "W6x9", "W6x15", "W6x25",
    "W8x10", "W8x18", "W8x24", "W8x31",
    "W10x12", "W10x19", "W10x26", "W10x33",
    "W12x14", "W12x19", "W12x26", "W12x35", "W12x50",
    "W14x22", "W14x30", "W14x43", "W14x61", "W14x82",
    "W16x26", "W16x36", "W16x50",
    "W18x35", "W18x50", "W18x55",
    "W21x44", "W21x62", "W21x68",
    "W24x55", "W24x76", "W24x84",
    "W27x84",
    "W30x90", "W30x108",
    "W33x118",
    "W36x135", "W36x150",
)

_AMERICAN_STANDARD = (
    "S4x7.7",
    "S6x12.5",
    "S8x18.4",
    "S10x25.4",
    "S12x31.8",
    "S15x42.9",
    "S18x54.7",
    "S24x80",
)

_CHANNELS = (
    "C3x4.1",
    "C4x5.4",
    "C5x6.7",
    "C6x8.2",
    "C6x10.5",
    "C7x9.8",
    "C8x11.5",
    "C8x13.75",
    "C9x13.4",
    "C10x15.3",
    "C12x20.7",
    "C12x30",
    "C15x33.9",
    "C15x40",
)

_ANGLES = (
    ("L2x2x1/8", 1.65),
    ("L2x2x3/16", 2.44),
    ("L2x2x1/4", 3.19),
    ("L2x2x5/16", 3.92),
    ("L2x2x3/8", 4.7),
    ("L3x3x3/16", 3.71),
    ("L3x3x1/4", 4.9),
    ("L3x3x5/16", 6.1),
    ("L3x3x3/8", 7.2),
    ("L3x3x1/2", 9.4),
    ("L4x3x1/4", 5.8),
    ("L4x3x3/8", 8.5),
    ("L4x3x1/2", 11.1),
    ("L4x4x1/4", 6.6),
    ("L4x4x5/16", 8.2),
    ("L4x4x3/8", 9.8),
    ("L4x4x1/2", 12.8),
    ("L5x3x1/4", 6.6),
    ("L5x3x3/8", 9.8),
    ("L5x3x1/2", 12.8),
    ("L5x5x3/8", 12.3),
    ("L5x5x1/2", 16.2),
    ("L6x4x3/8", 12.3),
    ("L6x4x1/2", 16.2),
    ("L6x6x3/8", 14.9),
    ("L6x6x1/2", 19.6),
    ("L6x6x3/4", 28.7),
    ("L8x4x1/2", 19.6),
    ("L8x6x1/2", 23.0),
    ("L8x8x1/2", 26.4),
    ("L8x8x3/4", 38.9),
)


STEEL_PROFILES: tuple[SteelProfile, ...] = (
    *(_rolled("wide_flange", name) for name in _WIDE_FLANGE),
    *(_rolled("american_standard", name) for name in _AMERICAN_STANDARD),
    *(_rolled("channel", name) for name in _CHANNELS),
    *(_angle(name, weight) for name, weight in _ANGLES),
)

# Nominal HSS weights, pounds per foot. Width, height, and wall are inches.
_HSS_ROWS = (
    (2, 2, 0.125, 3.05),
    (2, 2, 0.1875, 4.32),
    (2, 2, 0.25, 5.41),
    (3, 3, 0.125, 4.75),
    (3, 3, 0.1875, 6.87),
    (3, 3, 0.25, 8.81),
    (3, 3, 0.375, 12.17),
    (4, 4, 0.1875, 9.42),
    (4, 4, 0.25, 12.21),
    (4, 4, 0.375, 17.27),
    (4, 4, 0.5, 21.63),
    (5, 5, 0.25, 15.62),
    (5, 5, 0.375, 22.37),
    (6, 6, 0.25, 19.02),
    (6, 6, 0.375, 27.48),
    (6, 6, 0.5, 35.24),
    (8, 8, 0.25, 25.82),
    (8, 8, 0.375, 37.69),
    (8, 8, 0.5, 48.85),
    (10, 10, 0.375, 47.90),
    (10, 10, 0.5, 62.46),
    (12, 12, 0.5, 76.07),
    (3, 2, 0.1875, 5.59),
    (4, 2, 0.25, 8.81),
    (4, 3, 0.25, 10.51),
    (6, 4, 0.25, 15.62),
    (6, 4, 0.375, 22.37),
    (8, 4, 0.25, 19.02),
    (8, 4, 0.375, 27.48),
    (8, 6, 0.25, 22.42),
    (8, 6, 0.375, 32.58),
    (10, 6, 0.25, 25.82),
    (10, 6, 0.375, 37.69),
    (12, 6, 0.25, 29.23),
    (12, 6, 0.375, 42.79),
)

# Nominal size, schedule, outside diameter, wall, pounds per foot.
_PIPE_ROWS = (
    ("1", "40", 1.315, 0.133, 1.68),
    ("1-1/4", "40", 1.660, 0.140, 2.27),
    ("1-1/2", "40", 1.900, 0.145, 2.72),
    ("2", "40", 2.375, 0.154, 3.65),
    ("2-1/2", "40", 2.875, 0.203, 5.79),
    ("3", "40", 3.500, 0.216, 7.58),
    ("4", "40", 4.500, 0.237, 10.79),
    ("6", "40", 6.625, 0.280, 18.97),
    ("8", "40", 8.625, 0.322, 28.55),
    ("10", "40", 10.750, 0.365, 40.48),
    ("12", "40", 12.750, 0.406, 53.52),
    ("2", "80", 2.375, 0.218, 5.02),
    ("3", "80", 3.500, 0.300, 10.25),
    ("4", "80", 4.500, 0.337, 14.98),
    ("6", "80", 6.625, 0.432, 28.57),
)

TUBE_SIZES: tuple[TubeSize, ...] = (
    *(_hss(width, height, wall, weight) for width, height, wall, weight in _HSS_ROWS),
    *(_pipe(size, schedule, outside, wall, weight) for size, schedule, outside, wall, weight in _PIPE_ROWS),
)

PLATE_THICKNESSES: tuple[PlateThickness, ...] = (
    _plate("16 ga", 0.0598, "16ga", "16 gauge"),
    _plate("14 ga", 0.0747, "14ga", "14 gauge"),
    _plate("12 ga", 0.1046, "12ga", "12 gauge"),
    _plate("11 ga", 0.1196, "11ga", "11 gauge"),
    _plate("10 ga", 0.1345, "10ga", "10 gauge"),
    _plate("7 ga", 0.1793, "7ga", "7 gauge"),
    _plate("3/16", 0.1875, "3/16 in"),
    _plate("1/4", 0.25, "1/4 in"),
    _plate("5/16", 0.3125, "5/16 in"),
    _plate("3/8", 0.375, "3/8 in"),
    _plate("1/2", 0.5, "1/2 in"),
    _plate("5/8", 0.625, "5/8 in"),
    _plate("3/4", 0.75, "3/4 in"),
    _plate("1", 1.0, "1 in"),
    _plate("1-1/4", 1.25, "1 1/4", "1-1/4 in"),
    _plate("1-1/2", 1.5, "1 1/2", "1-1/2 in"),
    _plate("2", 2.0, "2 in"),
)

# Shop-fabrication planning bands. Not installed prices.
GATE_STYLES: tuple[GateStyle, ...] = (
    GateStyle(
        key="chain_link_single",
        name="Chain-link single swing",
        description="One swing leaf, chain-link fabric on a steel frame, standard hinges.",
        leaf_count=1,
        low_usd_per_sq_ft=22.0,
        high_usd_per_sq_ft=40.0,
        minimum_low_usd=550.0,
        minimum_high_usd=980.0,
        aliases=("chain link single", "single chain link", "single swing chain link"),
    ),
    GateStyle(
        key="chain_link_double",
        name="Chain-link double swing",
        description="Two swing leaves, chain-link fabric on steel frames, standard hinges.",
        leaf_count=2,
        low_usd_per_sq_ft=18.0,
        high_usd_per_sq_ft=34.0,
        minimum_low_usd=900.0,
        minimum_high_usd=1600.0,
        aliases=("chain link double", "double chain link", "double swing chain link"),
    ),
    GateStyle(
        key="ornamental_single",
        name="Ornamental single swing",
        description="One swing leaf with picket infill and standard hinges.",
        leaf_count=1,
        low_usd_per_sq_ft=48.0,
        high_usd_per_sq_ft=90.0,
        minimum_low_usd=800.0,
        minimum_high_usd=1500.0,
        aliases=("ornamental single", "picket single", "single ornamental"),
    ),
    GateStyle(
        key="ornamental_double",
        name="Ornamental double swing",
        description="Two swing leaves with picket infill and standard hinges.",
        leaf_count=2,
        low_usd_per_sq_ft=42.0,
        high_usd_per_sq_ft=80.0,
        minimum_low_usd=1400.0,
        minimum_high_usd=2600.0,
        aliases=("ornamental double", "double ornamental", "picket double"),
    ),
    GateStyle(
        key="vertical_bar_single",
        name="Vertical-bar single swing",
        description="One security swing leaf with vertical-bar infill and standard hinges.",
        leaf_count=1,
        low_usd_per_sq_ft=58.0,
        high_usd_per_sq_ft=110.0,
        minimum_low_usd=950.0,
        minimum_high_usd=1800.0,
        aliases=("vertical bar single", "security single", "single vertical bar"),
    ),
    GateStyle(
        key="vertical_bar_double",
        name="Vertical-bar double swing",
        description="Two security swing leaves with vertical-bar infill and standard hinges.",
        leaf_count=2,
        low_usd_per_sq_ft=52.0,
        high_usd_per_sq_ft=98.0,
        minimum_low_usd=1700.0,
        minimum_high_usd=3200.0,
        aliases=("vertical bar double", "double vertical bar", "security double"),
    ),
    GateStyle(
        key="pedestrian",
        name="Pedestrian swing",
        description="A walk gate with a steel frame and standard hinges.",
        leaf_count=1,
        low_usd_per_sq_ft=35.0,
        high_usd_per_sq_ft=70.0,
        minimum_low_usd=400.0,
        minimum_high_usd=750.0,
        aliases=("walk gate", "man gate", "pedestrian gate"),
    ),
    GateStyle(
        key="rolling_slide",
        name="Rolling slide",
        description="A ground-track slide gate with standard rollers.",
        leaf_count=1,
        low_usd_per_sq_ft=55.0,
        high_usd_per_sq_ft=105.0,
        minimum_low_usd=1800.0,
        minimum_high_usd=3400.0,
        aliases=("rolling gate", "v-track", "ground track", "slide gate"),
    ),
    GateStyle(
        key="cantilever_slide",
        name="Cantilever slide",
        description="A cantilever slide gate with no ground track across the opening.",
        leaf_count=1,
        low_usd_per_sq_ft=75.0,
        high_usd_per_sq_ft=145.0,
        minimum_low_usd=2500.0,
        minimum_high_usd=4800.0,
        aliases=("cantilever", "cantilever gate", "cantilever slide gate"),
    ),
)

# Shop-fabrication planning bands for one primed flight. Not installed prices.
# Straight treads use a 12 in depth (11 in run plus a 1 in nosing).
# Channel pans are a 3/16 allowance plus the nosing. Risers are 1/8 plate.
# A mono stringer is a center plate that deepens as the stair gets wider.
# Floating treads are 3/8 plate on cantilever arms. A spiral column matches
# 6 in Schedule 40 pipe, and its treads are 1/4 plate plus a formed edge.
STAIR_STYLES: tuple[StairStyle, ...] = (
    StairStyle(
        key="mono_stringer",
        name="Mono-stringer",
        description=(
            "One center plate stringer with plate treads on brackets and open risers. "
            "Handrail, guard, landing, and installation stay separate."
        ),
        layout="straight",
        stringer_count=1,
        stringer_lb_per_ft=8.0,
        stringer_lb_per_ft_per_in=0.35,
        tread_lb_per_sq_ft=10.20,
        tread_depth_in=12.0,
        riser_lb_per_sq_ft=0.0,
        connector_lb_per_step=8.0,
        low_usd_per_lb=6.00,
        high_usd_per_lb=11.00,
        minimum_low_usd=2200.0,
        minimum_high_usd=4200.0,
        aliases=("mono stringer", "center stringer", "single stringer"),
    ),
    StairStyle(
        key="channel_stringer",
        name="Channel stringer",
        description=(
            "Two channel stringers with pan treads and closed risers. "
            "Handrail, guard, landing, and installation stay separate."
        ),
        layout="straight",
        stringer_count=2,
        stringer_lb_per_ft=15.3,
        stringer_lb_per_ft_per_in=0.0,
        tread_lb_per_sq_ft=8.16,
        tread_depth_in=12.0,
        riser_lb_per_sq_ft=5.10,
        connector_lb_per_step=4.0,
        low_usd_per_lb=4.50,
        high_usd_per_lb=8.50,
        minimum_low_usd=1800.0,
        minimum_high_usd=3400.0,
        aliases=("channel stringers", "double stringer", "pan stair"),
    ),
    StairStyle(
        key="spiral",
        name="Spiral",
        description=(
            "A center pipe column with a plate tread at every step. "
            "Width is the overall diameter. Handrail, cage, landing, and installation stay separate."
        ),
        layout="spiral",
        stringer_count=1,
        stringer_lb_per_ft=18.97,
        stringer_lb_per_ft_per_in=0.0,
        tread_lb_per_sq_ft=11.73,
        tread_depth_in=0.0,
        riser_lb_per_sq_ft=0.0,
        connector_lb_per_step=8.0,
        low_usd_per_lb=9.00,
        high_usd_per_lb=16.00,
        minimum_low_usd=3200.0,
        minimum_high_usd=6000.0,
        aliases=("spiral stair", "spiral staircase", "circular stair"),
    ),
    StairStyle(
        key="floating",
        name="Floating stairs",
        description=(
            "One wall box stringer and cantilever plate treads with open risers. "
            "The wall, handrail, landing, and installation stay separate."
        ),
        layout="straight",
        stringer_count=1,
        stringer_lb_per_ft=18.0,
        stringer_lb_per_ft_per_in=0.15,
        tread_lb_per_sq_ft=15.30,
        tread_depth_in=12.0,
        riser_lb_per_sq_ft=0.0,
        connector_lb_per_step=14.0,
        low_usd_per_lb=8.00,
        high_usd_per_lb=15.00,
        minimum_low_usd=2800.0,
        minimum_high_usd=5400.0,
        aliases=("floating stair", "floating staircase", "cantilever stair"),
    ),
)

# Profiles are catalog designations. The first span that is at least the
# takeoff span wins, so a longer bay steps up to a heavier HSS.
_PATIO_POSTS: tuple[tuple[float, str], ...] = (
    (12.0, "HSS4x4x1/4"),
    (18.0, "HSS6x6x1/4"),
    (24.0, "HSS6x6x3/8"),
)
_PATIO_BEAMS: tuple[tuple[float, str], ...] = (
    (12.0, "HSS6x4x1/4"),
    (18.0, "HSS8x4x1/4"),
    (24.0, "HSS8x6x1/4"),
)
_PATIO_RAFTERS: tuple[tuple[float, str], ...] = (
    (10.0, "HSS3x3x1/4"),
    (16.0, "HSS4x4x1/4"),
    (22.0, "HSS6x4x1/4"),
    (24.0, "HSS6x6x1/4"),
)
_PATIO_LATTICE = "HSS2x2x1/8"
_PATIO_PURLIN = "HSS2x2x3/16"
_RAILING_POST = "HSS2x2x1/4"
_RAILING_RAIL = "HSS2x2x1/8"

# Shop-fabrication planning bands for one primed outdoor frame. Not installed prices.
PATIO_STYLES: tuple[PatioStyle, ...] = (
    PatioStyle(
        key="pergola",
        name="Steel pergola",
        description=(
            "An open HSS frame with posts, beams, rafters, and a light lattice. "
            "Roof panels, footings, and installation stay separate."
        ),
        kind="pergola",
        low_usd_per_lb=5.50,
        high_usd_per_lb=10.50,
        minimum_low_usd=2200.0,
        minimum_high_usd=4200.0,
        aliases=("steel pergola", "pergola"),
    ),
    PatioStyle(
        key="patio_cover",
        name="Patio cover",
        description=(
            "A roof-ready HSS frame with posts, beams, rafters, and purlins. "
            "Panels, gutters, flashing, footings, and installation stay separate."
        ),
        kind="cover",
        low_usd_per_lb=6.00,
        high_usd_per_lb=11.50,
        minimum_low_usd=2800.0,
        minimum_high_usd=5400.0,
        aliases=("patio cover", "patio roof", "cover"),
    ),
    PatioStyle(
        key="framing",
        name="Framing",
        description=(
            "Posts and beams only, for a cover or screen priced later. "
            "Footings and installation stay separate."
        ),
        kind="framing",
        low_usd_per_lb=4.50,
        high_usd_per_lb=8.50,
        minimum_low_usd=1600.0,
        minimum_high_usd=3000.0,
        aliases=("patio framing", "outdoor framing", "post and beam"),
    ),
    PatioStyle(
        key="perimeter_railing",
        name="Perimeter railing",
        description=(
            "Posts and three rails around the footprint. "
            "Infill panels and anchors stay separate."
        ),
        kind="railing",
        low_usd_per_lb=7.00,
        high_usd_per_lb=13.00,
        minimum_low_usd=900.0,
        minimum_high_usd=1700.0,
        aliases=("perimeter rail", "patio railing", "railing"),
    ),
)

# Shop-fabrication planning bands for one primed railing run. Not installed prices.
# Posts and rails are catalog HSS. Vertical and ornamental infill is 1/2 in
# square bar. Horizontal bars use the same bar. Cable infill is 1/8 in cable
# with an end fitting. The selected mount is added once per post.
RAILING_STYLES: tuple[RailingStyle, ...] = (
    RailingStyle(
        key="vertical_picket",
        name="Vertical picket",
        description=(
            "HSS posts and rails with vertical 1/2 in square-bar pickets at 4.5 in centers. "
            "A typical interior height is 36 in. Anchors and installation stay separate."
        ),
        kind="vertical",
        post="HSS2x2x1/8",
        top_rail="HSS2x2x1/8",
        bottom_rail="HSS2x2x1/8",
        post_spacing_ft=5.0,
        default_height_ft=3.0,
        infill="vertical_picket",
        low_usd_per_lb=6.50,
        high_usd_per_lb=12.00,
        minimum_low_usd=650.0,
        minimum_high_usd=1200.0,
        aliases=("vertical pickets", "picket railing", "vertical balustrade"),
    ),
    RailingStyle(
        key="horizontal_picket",
        name="Horizontal picket",
        description=(
            "HSS posts and rails with horizontal 1/2 in square bars at 4 in. "
            "A typical guard height is 42 in. Anchors and installation stay separate."
        ),
        kind="horizontal",
        post="HSS2x2x1/4",
        top_rail="HSS2x2x1/8",
        bottom_rail="HSS2x2x1/8",
        post_spacing_ft=5.0,
        default_height_ft=3.5,
        infill="horizontal_picket",
        low_usd_per_lb=7.00,
        high_usd_per_lb=13.00,
        minimum_low_usd=800.0,
        minimum_high_usd=1500.0,
        aliases=("horizontal pickets", "horizontal bar", "horizontal railing"),
    ),
    RailingStyle(
        key="cable",
        name="Cable railing",
        description=(
            "HSS posts and rails with horizontal 1/8 in cable and an end fitting on each run. "
            "Posts are closer because the cable is in tension. Glass and a wood cap stay separate."
        ),
        kind="cable",
        post="HSS2x2x1/4",
        top_rail="HSS2x2x1/8",
        bottom_rail="HSS2x2x1/8",
        post_spacing_ft=4.0,
        default_height_ft=3.5,
        infill="cable",
        low_usd_per_lb=8.00,
        high_usd_per_lb=15.00,
        minimum_low_usd=900.0,
        minimum_high_usd=1700.0,
        aliases=("cable rail", "cable railing", "horizontal cable"),
    ),
    RailingStyle(
        key="ornamental",
        name="Ornamental iron",
        description=(
            "Heavier HSS posts, a mid rail, and vertical 1/2 in square bars at 4 in. "
            "The closer bars stand in for ornamental infill. Anchors and installation stay separate."
        ),
        kind="ornamental",
        post="HSS2x2x1/4",
        top_rail="HSS2x2x1/4",
        bottom_rail="HSS2x2x3/16",
        post_spacing_ft=4.5,
        default_height_ft=3.5,
        infill="ornamental",
        low_usd_per_lb=8.00,
        high_usd_per_lb=15.00,
        minimum_low_usd=1000.0,
        minimum_high_usd=1900.0,
        aliases=("ornamental iron", "ornamental railing", "iron balustrade"),
    ),
)

RAILING_MOUNTS: tuple[RailingMount, ...] = (
    RailingMount(
        key="base_plate",
        name="Surface base plate",
        description="A 3/8 in plate, 6 in by 6 in, at each post. Anchors stay separate.",
        kind="plate",
        stock="3/8",
        width_in=6.0,
        length_in=6.0,
        length_ft=0.0,
        aliases=("base plate", "surface mount", "floor plate"),
    ),
    RailingMount(
        key="fascia_plate",
        name="Fascia plate",
        description="A 3/8 in side plate, 4 in by 8 in, at each post. Anchors stay separate.",
        kind="plate",
        stock="3/8",
        width_in=4.0,
        length_in=8.0,
        length_ft=0.0,
        aliases=("fascia", "side mount", "fascia mount"),
    ),
    RailingMount(
        key="wall_bracket",
        name="Wall bracket",
        description="A 6 in piece of L2x2x1/4 at each post. Anchors stay separate.",
        kind="member",
        stock="L2x2x1/4",
        width_in=0.0,
        length_in=0.0,
        length_ft=0.5,
        aliases=("wall mount", "bracket", "angle bracket"),
    ),
    RailingMount(
        key="core_sleeve",
        name="Core sleeve",
        description="A 6 in piece of 2 in Schedule 40 pipe at each post. Core drilling stays separate.",
        kind="member",
        stock="PIPE 2 SCH 40",
        width_in=0.0,
        length_in=0.0,
        length_ft=0.5,
        aliases=("core mount", "sleeve", "core drill sleeve"),
    ),
)


@dataclass(frozen=True)
class Upgrade:
    """A professional add-on priced on its own line.

    ``per_lb`` rates are USD per pound of the takeoff, with a shop minimum.
    ``added_steel`` rates are fractions of the base bracket and cover only
    the extra steel for a heavier member. Neither rate is inside the base bracket.
    """

    key: str
    name: str
    method: str
    low_rate: float
    high_rate: float
    minimum_low_usd: float
    minimum_high_usd: float
    families: tuple[str, ...]
    details: Mapping[str, str]


@dataclass(frozen=True)
class UpgradeOffer:
    """One suggested add-on for a takeoff that has already been priced."""

    key: str
    name: str
    family: str
    detail: str
    low_usd: float
    high_usd: float

    @property
    def label(self) -> str:
        """Return the add-on bracket, such as ``$450.00–$800.00``."""
        return f"${self.low_usd:,.2f}–${self.high_usd:,.2f}"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dictionary without a single price."""
        return {
            "key": self.key,
            "name": self.name,
            "family": self.family,
            "detail": self.detail,
            "low_usd": self.low_usd,
            "high_usd": self.high_usd,
            "range": self.label,
        }


_PROJECT_FAMILIES = ("gate", "stair", "patio", "railing", "structural")

# Suggested after the base bracket. Order is the order shown for that project.
ADDON_ORDER: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "gate": ("hot_dip_galvanizing", "marine_powder", "heavy_gauge"),
        "stair": ("hot_dip_galvanizing", "heavy_gauge", "marine_powder"),
        "patio": ("hot_dip_galvanizing", "heavy_gauge", "marine_powder"),
        "railing": ("hot_dip_galvanizing", "heavy_gauge", "marine_powder"),
        "structural": ("heavy_gauge", "hot_dip_galvanizing", "marine_powder"),
    }
)

UPGRADES: tuple[Upgrade, ...] = (
    Upgrade(
        key="hot_dip_galvanizing",
        name="Hot-dip galvanizing",
        method="per_lb",
        low_rate=0.55,
        high_rate=0.95,
        minimum_low_usd=450.0,
        minimum_high_usd=800.0,
        families=_PROJECT_FAMILIES,
        details=MappingProxyType(
            {
                "gate": (
                    "Galvanize an exterior leaf after fabrication and before any color coat. "
                    "The kettle charge stays off the fabrication bracket."
                ),
                "stair": (
                    "Galvanize an exterior flight after the shop weld. "
                    "An interior flight can stay primed. Keep the kettle charge off the flight bracket."
                ),
                "patio": (
                    "Hot-dip the outdoor frame after fabrication. Powder alone will not protect the joints. "
                    "Keep the kettle charge off the frame bracket."
                ),
                "railing": (
                    "Hot-dip an exterior railing or balcony guard after fabrication. "
                    "An interior railing can stay primed. Keep the kettle charge off the railing bracket."
                ),
                "structural": (
                    "If this steel will be exterior, galvanize it after fabrication. "
                    "Keep that charge off the material bracket."
                ),
            }
        ),
    ),
    Upgrade(
        key="heavy_gauge",
        name="Heavy-gauge structural framing",
        method="added_steel",
        low_rate=0.15,
        high_rate=0.30,
        minimum_low_usd=0.0,
        minimum_high_usd=0.0,
        families=_PROJECT_FAMILIES,
        details=MappingProxyType(
            {
                "gate": (
                    "For a vehicle leaf or a tall opening, price the next heavier frame tube. "
                    "This band is the added steel, not a second copy of the leaf."
                ),
                "stair": (
                    "Step the stringer or the tread plate up one thickness when the flight carries more than a household load. "
                    "This band is only the added steel."
                ),
                "patio": (
                    "On a long span or a snow load, price the next heavier post and beam wall. "
                    "This band is the added steel for that step-up."
                ),
                "railing": (
                    "On a long balcony or a crowd load, price the next heavier post. "
                    "This band is the added steel, not a second run."
                ),
                "structural": (
                    "Price the next heavier profile or the next wall before the order is cut. "
                    "This band is the added steel, not a second stick of the same size."
                ),
            }
        ),
    ),
    Upgrade(
        key="marine_powder",
        name="Marine-grade powder coating",
        method="per_lb",
        low_rate=0.40,
        high_rate=0.80,
        minimum_low_usd=500.0,
        minimum_high_usd=950.0,
        families=_PROJECT_FAMILIES,
        details=MappingProxyType(
            {
                "gate": (
                    "Use a marine-grade powder over the galvanizing for a coastal, poolside, or finished entry leaf. "
                    "The color coat does not replace the zinc."
                ),
                "stair": (
                    "Use a marine-grade powder over galvanizing when the flight is exposed. "
                    "A primed interior flight does not need it."
                ),
                "patio": (
                    "Marine-grade powder over the galvanizing is the color coat for a coastal or highly visible frame."
                ),
                "railing": (
                    "Use a marine-grade powder over galvanizing when the railing is coastal, poolside, or a finished balcony. "
                    "An interior primed rail does not need it."
                ),
                "structural": (
                    "Add marine-grade powder only when the member is an exposed finish. "
                    "Outdoors it goes over galvanizing, and it stays off the material bracket."
                ),
            }
        ),
    ),
)


def _normalize_shape(value: str) -> str:
    """Fold a profile or tube name to a lookup key."""
    text = value.strip().lower()
    text = text.replace("×", "x").replace("–", "-").replace("—", "-").replace("″", "").replace('"', "")
    text = re.sub(r"(\d)\s+(\d+/\d+)", r"\1-\2", text)
    return re.sub(r"\s+", "", text)


def _normalize_plate(value: str) -> str:
    """Fold a plate name to a lookup key."""
    text = _normalize_shape(value)
    text = text.replace("gauge", "ga").replace("gage", "ga")
    return re.sub(r"(inches|inch|in)$", "", text)


def _normalize_gate(value: str) -> str:
    """Fold a gate or stair style name to a lookup key."""
    text = value.strip().lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace(" ", "_")


def _index_members() -> dict[str, SteelProfile | TubeSize]:
    """Index profiles and tubes by every accepted name."""
    index: dict[str, SteelProfile | TubeSize] = {}
    for item in (*STEEL_PROFILES, *TUBE_SIZES):
        names = [item.designation]
        if isinstance(item, TubeSize):
            names.extend(item.aliases)
        for name in names:
            key = _normalize_shape(name)
            current = index.get(key)
            if current is not None and current is not item:
                raise RuntimeError(f"Duplicate material name {name!r}.")
            index[key] = item
    return index


def _index_plates() -> dict[str, PlateThickness]:
    """Index plate rows by name, alias, and exact decimal thickness."""
    index: dict[str, PlateThickness] = {}

    def add(key: str, plate: PlateThickness) -> None:
        current = index.get(key)
        if current is not None and current is not plate:
            raise RuntimeError(f"Duplicate plate name {key!r}.")
        index[key] = plate

    for plate in PLATE_THICKNESSES:
        add(_normalize_plate(plate.name), plate)
        for alias in plate.aliases:
            add(_normalize_plate(alias), plate)
        rendered = f"{plate.thickness_in:.4f}".rstrip("0").rstrip(".")
        add(rendered, plate)
        add(f"{plate.thickness_in:.4f}", plate)
    return index


def _index_gates() -> dict[str, GateStyle]:
    """Index gate styles by key and alias."""
    index: dict[str, GateStyle] = {}
    for style in GATE_STYLES:
        for name in (style.key, style.name, *style.aliases):
            key = _normalize_gate(name)
            current = index.get(key)
            if current is not None and current is not style:
                raise RuntimeError(f"Duplicate gate style {name!r}.")
            index[key] = style
    return index


def _index_stairs() -> dict[str, StairStyle]:
    """Index stair styles by key and alias."""
    index: dict[str, StairStyle] = {}
    for style in STAIR_STYLES:
        for name in (style.key, style.name, *style.aliases):
            key = _normalize_gate(name)
            current = index.get(key)
            if current is not None and current is not style:
                raise RuntimeError(f"Duplicate stair style {name!r}.")
            index[key] = style
    return index


def _index_railing_mounts() -> dict[str, RailingMount]:
    """Index railing mounts by key and alias."""
    index: dict[str, RailingMount] = {}
    for mount in RAILING_MOUNTS:
        for name in (mount.key, mount.name, *mount.aliases):
            key = _normalize_gate(name)
            current = index.get(key)
            if current is not None and current is not mount:
                raise RuntimeError(f"Duplicate railing mount {name!r}.")
            index[key] = mount
    return index


def _index_railings() -> dict[str, RailingStyle]:
    """Index railing styles by key and alias."""
    index: dict[str, RailingStyle] = {}
    for style in RAILING_STYLES:
        for name in (style.key, style.name, *style.aliases):
            key = _normalize_gate(name)
            current = index.get(key)
            if current is not None and current is not style:
                raise RuntimeError(f"Duplicate railing style {name!r}.")
            index[key] = style
    return index


def _index_patios() -> dict[str, PatioStyle]:
    """Index patio structures by key and alias."""
    index: dict[str, PatioStyle] = {}
    for style in PATIO_STYLES:
        for name in (style.key, style.name, *style.aliases):
            key = _normalize_gate(name)
            current = index.get(key)
            if current is not None and current is not style:
                raise RuntimeError(f"Duplicate patio structure {name!r}.")
            index[key] = style
    return index


def _validate_catalog() -> None:
    """Reject a catalog whose weights or price bands contradict themselves."""
    if not MATERIAL_LOW_USD_PER_LB < MATERIAL_HIGH_USD_PER_LB:
        raise RuntimeError("The material rate band is inverted.")
    if HIGH_SIDE_ALLOWANCE < 0:
        raise RuntimeError("HIGH_SIDE_ALLOWANCE cannot be negative.")
    for profile in STEEL_PROFILES:
        if profile.weight_lb_per_ft <= 0:
            raise RuntimeError(f"{profile.designation} has no weight.")
        if profile.family != "angle":
            nominal = float(profile.designation.split("x")[-1])
            if not math.isclose(profile.weight_lb_per_ft, nominal, rel_tol=0, abs_tol=1e-9):
                raise RuntimeError(f"{profile.designation} weight does not match its designation.")
    for tube in TUBE_SIZES:
        if tube.weight_lb_per_ft <= 0 or tube.wall_in <= 0:
            raise RuntimeError(f"{tube.designation} has no weight.")
        if tube.shape == "square" and tube.outside_width_in != tube.outside_height_in:
            raise RuntimeError(f"{tube.designation} is marked square but the sides differ.")
    for plate in PLATE_THICKNESSES:
        expected = round(plate.thickness_in * STEEL_LB_PER_SQ_FT_PER_INCH, 2)
        if plate.weight_lb_per_sq_ft != expected:
            raise RuntimeError(f"{plate.name} weight does not match {STEEL_LB_PER_SQ_FT_PER_INCH} lb/ft²/in.")
    for style in GATE_STYLES:
        ordered = (
            0 < style.low_usd_per_sq_ft < style.high_usd_per_sq_ft
            and 0 < style.minimum_low_usd < style.minimum_high_usd
            and style.leaf_count >= 1
        )
        if not ordered:
            raise RuntimeError(f"{style.key} has an inverted price band.")
    for style in STAIR_STYLES:
        ordered = (
            style.layout in {"straight", "spiral"}
            and style.stringer_count >= 1
            and style.stringer_lb_per_ft > 0
            and style.stringer_lb_per_ft_per_in >= 0
            and style.tread_lb_per_sq_ft > 0
            and style.riser_lb_per_sq_ft >= 0
            and style.connector_lb_per_step >= 0
            and 0 < style.low_usd_per_lb < style.high_usd_per_lb
            and 0 < style.minimum_low_usd < style.minimum_high_usd
            and (style.layout == "spiral" or style.tread_depth_in > 0)
        )
        if not ordered:
            raise RuntimeError(f"{style.key} has an inverted stair band or weight factor.")
    patio_kinds = {"pergola", "cover", "framing", "railing"}
    for style in PATIO_STYLES:
        ordered = (
            style.kind in patio_kinds
            and 0 < style.low_usd_per_lb < style.high_usd_per_lb
            and 0 < style.minimum_low_usd < style.minimum_high_usd
        )
        if not ordered:
            raise RuntimeError(f"{style.key} has an inverted patio band.")
    for designation in (
        *(name for _limit, name in (*_PATIO_POSTS, *_PATIO_BEAMS, *_PATIO_RAFTERS)),
        _PATIO_LATTICE,
        _PATIO_PURLIN,
        _RAILING_POST,
        _RAILING_RAIL,
    ):
        if _normalize_shape(designation) not in _MEMBERS:
            raise RuntimeError(f"{designation} is used for a patio frame but is not in the catalog.")
    railing_kinds = {"vertical", "horizontal", "cable", "ornamental"}
    plate_names = {plate.name for plate in PLATE_THICKNESSES}
    for style in RAILING_STYLES:
        ordered = (
            style.kind in railing_kinds
            and style.infill in {"vertical_picket", "horizontal_picket", "cable", "ornamental"}
            and style.post_spacing_ft > 0
            and _MIN_RAILING_HEIGHT_FT <= style.default_height_ft <= _MAX_RAILING_HEIGHT_FT
            and 0 < style.low_usd_per_lb < style.high_usd_per_lb
            and 0 < style.minimum_low_usd < style.minimum_high_usd
        )
        if not ordered:
            raise RuntimeError(f"{style.key} has an inverted railing band.")
        for designation in (style.post, style.top_rail, style.bottom_rail):
            if _normalize_shape(designation) not in _MEMBERS:
                raise RuntimeError(f"{designation} is used for a railing but is not in the catalog.")
    for mount in RAILING_MOUNTS:
        if mount.kind == "plate":
            valid = mount.stock in plate_names and mount.width_in > 0 and mount.length_in > 0
        elif mount.kind == "member":
            valid = _normalize_shape(mount.stock) in _MEMBERS and mount.length_ft > 0
        else:
            valid = False
        if not valid:
            raise RuntimeError(f"{mount.key} is not a usable railing mount.")
    seen_upgrades: set[str] = set()
    for upgrade in UPGRADES:
        if upgrade.key in seen_upgrades:
            raise RuntimeError(f"Duplicate upgrade {upgrade.key}.")
        seen_upgrades.add(upgrade.key)
        if upgrade.method not in {"per_lb", "added_steel"}:
            raise RuntimeError(f"{upgrade.key} has an unknown pricing method.")
        if not 0 < upgrade.low_rate < upgrade.high_rate:
            raise RuntimeError(f"{upgrade.key} has an inverted rate.")
        if upgrade.method == "per_lb" and not 0 < upgrade.minimum_low_usd < upgrade.minimum_high_usd:
            raise RuntimeError(f"{upgrade.key} has an inverted shop minimum.")
        if upgrade.method == "added_steel" and upgrade.high_rate >= 1:
            raise RuntimeError(f"{upgrade.key} would replace the base bracket instead of adding steel.")
        if tuple(upgrade.details) != upgrade.families:
            raise RuntimeError(f"{upgrade.key} details do not match its project types.")
        if any(family not in _PROJECT_FAMILIES for family in upgrade.families):
            raise RuntimeError(f"{upgrade.key} names an unknown project type.")
    for family, keys in ADDON_ORDER.items():
        if family not in _PROJECT_FAMILIES or tuple(sorted(keys)) != tuple(sorted(seen_upgrades)):
            raise RuntimeError(f"{family} does not list every upgrade.")


_MEMBERS = _index_members()
_PLATES = _index_plates()
_GATES = _index_gates()
_STAIRS = _index_stairs()
_PATIOS = _index_patios()
_RAILINGS = _index_railings()
_RAILING_MOUNTS = _index_railing_mounts()
_validate_catalog()

STRUCTURAL_STEEL_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        item.designation: item.weight_lb_per_ft
        for item in (*STEEL_PROFILES, *TUBE_SIZES)
    }
)


def _require_text(value: str, name: str) -> str:
    """Return a stripped string or raise when the caller sent a blank value."""
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{name} must be a non-empty string.")
    return value.strip()


def _positive_number(value: object, name: str, maximum: float) -> float:
    """Return a finite number inside (0, maximum]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogError(f"{name} must be a number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise CatalogError(f"{name} must be a positive number.")
    if number > maximum:
        raise CatalogError(f"{name} must be {maximum:g} or less.")
    return number


def _quantity(value: int) -> int:
    """Return a piece count of at least one."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CatalogError("quantity must be an integer of at least 1.")
    if value > 10_000:
        raise CatalogError("quantity must be 10000 or less.")
    return value


def _step_count(value: object) -> int:
    """Return a stair step count inside the planning range."""
    if isinstance(value, bool):
        raise CatalogError("step_count must be an integer.")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise CatalogError("step_count must be an integer.")
    if value < _MIN_STEPS or value > _MAX_STEPS:
        raise CatalogError(f"step_count must be from {_MIN_STEPS} to {_MAX_STEPS}.")
    return value


def _bounded_number(value: object, name: str, low: float, high: float) -> float:
    """Return a finite number inside [low, high]."""
    number = _positive_number(value, name, high)
    if number < low:
        raise CatalogError(f"{name} must be at least {low:g}.")
    return number


def lookup_profile(designation: str) -> SteelProfile:
    """Return a rolled profile by designation.

    Args:
        designation: A catalog name such as ``W12x26`` or ``L4x4x1/4``.

    Returns:
        The matching profile.

    Raises:
        CatalogError: The name is blank, unknown, or names a tube.
    """
    found = _lookup_member(_require_text(designation, "designation"))
    if isinstance(found, TubeSize):
        raise CatalogError(f"{found.designation} is a tube. Look it up with lookup_tube.")
    return found


def lookup_tube(designation: str) -> TubeSize:
    """Return an HSS or pipe size by designation.

    Args:
        designation: A catalog name such as ``HSS4x4x1/4`` or ``2 SCH 40``.

    Returns:
        The matching tube.

    Raises:
        CatalogError: The name is blank, unknown, or names a rolled profile.
    """
    found = _lookup_member(_require_text(designation, "designation"))
    if isinstance(found, SteelProfile):
        raise CatalogError(f"{found.designation} is a rolled profile. Look it up with lookup_profile.")
    return found


def lookup_plate(name: str) -> PlateThickness:
    """Return a plate thickness by gauge, fraction, or decimal inches.

    Args:
        name: A catalog name such as ``1/2``, ``10 ga``, or ``0.25``.

    Returns:
        The matching thickness.

    Raises:
        CatalogError: The name is blank or not in the plate list.
    """
    key = _normalize_plate(_require_text(name, "plate"))
    found = _PLATES.get(key)
    if found is None:
        raise CatalogError(f"No plate thickness named {name.strip()!r}.")
    return found


def lookup_gate(style: str) -> GateStyle:
    """Return a gate style by key or display name.

    Args:
        style: A catalog key such as ``cantilever_slide`` or ``cantilever slide``.

    Returns:
        The matching style.

    Raises:
        CatalogError: The name is blank or not in the gate list.
    """
    key = _normalize_gate(_require_text(style, "gate_style"))
    found = _GATES.get(key)
    if found is None:
        names = ", ".join(item.key for item in GATE_STYLES)
        raise CatalogError(f"No gate style named {style.strip()!r}. Known styles: {names}.")
    return found


def lookup_stair(style: str) -> StairStyle:
    """Return a stair style by key or display name.

    Args:
        style: A catalog key such as ``mono_stringer`` or ``mono-stringer``.

    Returns:
        The matching style.

    Raises:
        CatalogError: The name is blank or not in the stair list.
    """
    key = _normalize_gate(_require_text(style, "stair_style"))
    found = _STAIRS.get(key)
    if found is None:
        names = ", ".join(item.key for item in STAIR_STYLES)
        raise CatalogError(f"No stair style named {style.strip()!r}. Known styles: {names}.")
    return found


def lookup_patio(style: str) -> PatioStyle:
    """Return a patio structure by key or display name.

    Args:
        style: A catalog key such as ``pergola`` or ``steel pergola``.

    Returns:
        The matching structure.

    Raises:
        CatalogError: The name is blank or not in the patio list.
    """
    key = _normalize_gate(_require_text(style, "patio_style"))
    found = _PATIOS.get(key)
    if found is None:
        names = ", ".join(item.key for item in PATIO_STYLES)
        raise CatalogError(f"No patio structure named {style.strip()!r}. Known structures: {names}.")
    return found


def lookup_railing(style: str) -> RailingStyle:
    """Return a railing style by key or display name.

    Args:
        style: A catalog key such as ``cable`` or ``balcony guard``.

    Returns:
        The matching railing style.

    Raises:
        CatalogError: The name is blank or not in the railing list.
    """
    key = _normalize_gate(_require_text(style, "railing_style"))
    found = _RAILINGS.get(key)
    if found is None:
        names = ", ".join(item.name for item in RAILING_STYLES)
        raise CatalogError(f"No railing style named {style.strip()!r}. Known styles: {names}.")
    return found


def lookup_railing_mount(mount: str) -> RailingMount:
    """Return a railing mount by key or display name.

    Args:
        mount: A catalog key such as ``base_plate`` or ``fascia plate``.

    Returns:
        The matching mount.

    Raises:
        CatalogError: The name is blank or not in the mount list.
    """
    key = _normalize_gate(_require_text(mount, "railing_mount"))
    found = _RAILING_MOUNTS.get(key)
    if found is None:
        names = ", ".join(item.name for item in RAILING_MOUNTS)
        raise CatalogError(f"No railing mount named {mount.strip()!r}. Known mounts: {names}.")
    return found


def _lookup_member(designation: str) -> SteelProfile | TubeSize:
    """Return a profile or tube, with a hint when the name is unknown."""
    found = _MEMBERS.get(_normalize_shape(designation))
    if found is None:
        raise CatalogError(
            f"No profile or tube named {designation!r}. "
            "Use a designation such as W12x26, L4x4x1/4, HSS4x4x1/4, or 2 SCH 40."
        )
    return found


def member_weight_lb(designation: str, length_ft: float, quantity: int = 1) -> float:
    """Return the nominal weight of a profile or tube.

    Args:
        designation: Rolled shape, HSS, or pipe name.
        length_ft: Length of one piece, in feet.
        quantity: Number of pieces.

    Returns:
        Total nominal weight in pounds.

    Raises:
        CatalogError: The name is unknown or a dimension is out of range.
    """
    shape = _lookup_member(_require_text(designation, "designation"))
    length = _positive_number(length_ft, "length_ft", _MAX_LENGTH_FT)
    count = _quantity(quantity)
    weight = shape.weight_lb_per_ft * length * count
    if weight > _MAX_WEIGHT_LB:
        raise CatalogError(f"The computed weight exceeds {_MAX_WEIGHT_LB:g} lb.")
    return weight


def plate_weight_lb(thickness_in: float, width_in: float, length_in: float, quantity: int = 1) -> float:
    """Return the nominal weight of a rectangular plate.

    Args:
        thickness_in: Plate thickness in inches.
        width_in: Plate width in inches.
        length_in: Plate length in inches.
        quantity: Number of plates.

    Returns:
        Total nominal weight in pounds.

    Raises:
        CatalogError: A dimension is out of range.
    """
    thickness = _positive_number(thickness_in, "thickness_in", _MAX_THICKNESS_IN)
    width = _positive_number(width_in, "width_in", _MAX_PLATE_SIDE_IN)
    length = _positive_number(length_in, "length_in", _MAX_PLATE_SIDE_IN)
    count = _quantity(quantity)
    area_sq_ft = (width / 12.0) * (length / 12.0)
    weight = area_sq_ft * thickness * STEEL_LB_PER_SQ_FT_PER_INCH * count
    if weight > _MAX_WEIGHT_LB:
        raise CatalogError(f"The computed weight exceeds {_MAX_WEIGHT_LB:g} lb.")
    return weight


def stair_takeoff(
    style: str,
    step_count: int,
    width_in: float,
    *,
    rise_in: float | None = None,
    run_in: float | None = None,
    quantity: int = 1,
) -> StairTakeoff:
    """Return the planning steel weight for a custom stair flight.

    Straight stairs use one less fabricated tread than the step count,
    because the upper floor is the landing. A spiral fabricates one tread
    per step. Rise and run fall back to the catalog defaults.

    Args:
        style: Stair catalog key or display name.
        step_count: Number of risers to climb.
        width_in: Overall width in inches. For a spiral, this is the diameter.
        rise_in: Riser height in inches. Defaults to ``STAIR_DEFAULT_RISE_IN``.
        run_in: Tread run in inches for a straight stair. Defaults to
            ``STAIR_DEFAULT_RUN_IN``. A spiral does not use it.
        quantity: Number of flights.

    Returns:
        Component weights for one flight and the total for the quantity.

    Raises:
        CatalogError: The style is unknown or a dimension is out of range.
    """
    chosen = lookup_stair(style)
    steps = _step_count(step_count)
    count = _quantity(quantity)
    rise = STAIR_DEFAULT_RISE_IN if rise_in is None else _bounded_number(rise_in, "rise_in", _MIN_RISE_IN, _MAX_RISE_IN)
    if chosen.layout == "spiral":
        width = _bounded_number(width_in, "stair_width_in", _MIN_SPIRAL_DIAMETER_IN, _MAX_SPIRAL_DIAMETER_IN)
        run = None
        height_ft = steps * rise / 12.0
        radius_in = width / 2.0
        inner_in = STAIR_SPIRAL_INNER_RADIUS_IN
        sector = STAIR_SPIRAL_DEGREES_PER_STEP / 360.0
        tread_area = math.pi * (radius_in**2 - inner_in**2) * sector / 144.0
        stringer_lb = chosen.stringer_count * height_ft * (
            chosen.stringer_lb_per_ft + chosen.stringer_lb_per_ft_per_in * width
        )
        tread_lb = steps * tread_area * chosen.tread_lb_per_sq_ft
        riser_lb = 0.0
    else:
        width = _bounded_number(width_in, "stair_width_in", _MIN_STAIR_WIDTH_IN, _MAX_STAIR_WIDTH_IN)
        run = STAIR_DEFAULT_RUN_IN if run_in is None else _bounded_number(run_in, "run_in", _MIN_RUN_IN, _MAX_RUN_IN)
        slope_ft = steps * math.hypot(rise, run) / 12.0
        tread_count = max(steps - 1, 1)
        tread_area = (width / 12.0) * (chosen.tread_depth_in / 12.0)
        stringer_lb = chosen.stringer_count * slope_ft * (
            chosen.stringer_lb_per_ft + chosen.stringer_lb_per_ft_per_in * width
        )
        tread_lb = tread_count * tread_area * chosen.tread_lb_per_sq_ft
        riser_area = (width / 12.0) * (rise / 12.0)
        riser_lb = steps * riser_area * chosen.riser_lb_per_sq_ft
    connector_lb = steps * chosen.connector_lb_per_step
    takeoff = StairTakeoff(
        style=chosen,
        step_count=steps,
        width_in=width,
        rise_in=rise,
        run_in=run,
        stringer_lb=stringer_lb,
        tread_lb=tread_lb,
        riser_lb=riser_lb,
        connector_lb=connector_lb,
        quantity=count,
    )
    if takeoff.weight_lb > _MAX_WEIGHT_LB:
        raise CatalogError(f"The computed weight exceeds {_MAX_WEIGHT_LB:g} lb.")
    return takeoff


def _pick_patio_profile(table: tuple[tuple[float, str], ...], span_ft: float) -> str:
    """Return the catalog designation for this span."""
    for limit, designation in table:
        if span_ft <= limit:
            return designation
    return table[-1][1]


def _patio_line(role: str, designation: str, count: int, length_ft: float) -> PatioLine:
    """Build one profile line from a catalog weight."""
    shape = _lookup_member(designation)
    return PatioLine(
        role=role,
        designation=shape.designation,
        count=count,
        length_ft=length_ft,
        weight_lb=shape.weight_lb_per_ft * length_ft * count,
    )


def _end_count(length_ft: float, spacing_ft: float) -> int:
    """Return how many members cross a length, including both ends."""
    return max(2, math.ceil(length_ft / spacing_ft) + 1)


def _infill_count(length_ft: float, spacing_ft: float) -> int:
    """Return how many members sit between the ends of a length."""
    return max(0, math.ceil(length_ft / spacing_ft) - 1)


def patio_takeoff(
    style: str,
    area_sq_ft: float,
    span_ft: float,
    *,
    quantity: int = 1,
) -> PatioTakeoff:
    """Return the catalog profiles and planning weight for an outdoor frame.

    Span is the rafter direction. The other side is the area divided by the
    span. Posts are about 10 ft apart. A span over 16 ft adds a center beam
    line. A pergola spaces rafters at 4 ft and adds a lattice. A patio cover
    spaces rafters at 2 ft and adds purlins. Framing is posts and beams only.
    A perimeter railing is posts at 6 ft and three rails around the footprint.

    Args:
        style: Patio catalog key or display name.
        area_sq_ft: Footprint area in square feet.
        span_ft: Primary span in feet.
        quantity: Number of structures.

    Returns:
        Named profile lines and the total weight.

    Raises:
        CatalogError: The style is unknown or the footprint is out of range.
    """
    chosen = lookup_patio(style)
    area = _bounded_number(area_sq_ft, "area_sq_ft", _MIN_PATIO_AREA_SQ_FT, _MAX_PATIO_AREA_SQ_FT)
    span = _bounded_number(span_ft, "span_ft", _MIN_PATIO_SPAN_FT, _MAX_PATIO_SPAN_FT)
    count = _quantity(quantity)
    width = area / span
    if width < _MIN_PATIO_WIDTH_FT or width > _MAX_PATIO_WIDTH_FT:
        raise CatalogError(
            f"A {span:g} ft span and {area:g} sq ft make a {width:g} ft width. "
            f"The width must be from {_MIN_PATIO_WIDTH_FT:g} to {_MAX_PATIO_WIDTH_FT:g} ft."
        )
    if chosen.kind == "railing":
        perimeter = 2.0 * (width + span)
        posts = max(4, math.ceil(perimeter / PATIO_RAILING_POST_SPACING_FT))
        lines = (
            _patio_line("posts", _RAILING_POST, posts, PATIO_RAILING_HEIGHT_FT),
            _patio_line("rails", _RAILING_RAIL, 3, perimeter),
        )
    else:
        post_lines = 2 if span <= 16 else 3
        posts = _end_count(width, PATIO_POST_SPACING_FT) * post_lines
        lines_list = [
            _patio_line("posts", _pick_patio_profile(_PATIO_POSTS, span), posts, PATIO_POST_HEIGHT_FT),
            _patio_line("beams", _pick_patio_profile(_PATIO_BEAMS, span), post_lines, width),
        ]
        if chosen.kind in {"pergola", "cover"}:
            spacing = 4.0 if chosen.kind == "pergola" else 2.0
            lines_list.append(
                _patio_line(
                    "rafters",
                    _pick_patio_profile(_PATIO_RAFTERS, span),
                    _end_count(width, spacing),
                    span,
                )
            )
        if chosen.kind == "pergola":
            lattice = _infill_count(span, 2.0)
            if lattice:
                lines_list.append(_patio_line("lattice", _PATIO_LATTICE, lattice, width))
        if chosen.kind == "cover":
            purlins = _infill_count(span, 4.0)
            if purlins:
                lines_list.append(_patio_line("purlins", _PATIO_PURLIN, purlins, width))
        lines = tuple(lines_list)
    takeoff = PatioTakeoff(
        style=chosen,
        area_sq_ft=area,
        span_ft=span,
        width_ft=width,
        lines=lines,
        quantity=count,
    )
    if takeoff.weight_lb > _MAX_WEIGHT_LB:
        raise CatalogError(f"The computed weight exceeds {_MAX_WEIGHT_LB:g} lb.")
    return takeoff


def _railing_member(role: str, designation: str, count: int, length_ft: float) -> RailingLine:
    """Build one railing line from a catalog weight."""
    shape = _lookup_member(designation)
    return RailingLine(
        role=role,
        designation=shape.designation,
        count=count,
        length_ft=length_ft,
        weight_lb=shape.weight_lb_per_ft * length_ft * count,
    )


def _railing_hardware(mount: RailingMount, count: int) -> RailingLine:
    """Build one piece of the selected hardware for every post."""
    if mount.kind == "plate":
        plate = lookup_plate(mount.stock)
        each = plate.weight_lb_per_sq_ft * (mount.width_in / 12.0) * (mount.length_in / 12.0)
        label = f"{mount.name}, {_plate_label(plate.name)} plate, {mount.width_in:g} in x {mount.length_in:g} in"
        return RailingLine(role="mounts", designation=label, count=count, length_ft=0.0, weight_lb=each * count)
    shape = _lookup_member(mount.stock)
    return RailingLine(
        role="mounts",
        designation=f"{mount.name}, {shape.designation}",
        count=count,
        length_ft=mount.length_ft,
        weight_lb=shape.weight_lb_per_ft * mount.length_ft * count,
    )


def railing_takeoff(
    style: str,
    length_ft: float,
    height_ft: float,
    *,
    mount: str = RAILING_DEFAULT_MOUNT,
    quantity: int = 1,
) -> RailingTakeoff:
    """Return the catalog steel, infill, and mounts for one railing run.

    Posts include both ends of the run. Vertical pickets and ornamental bars
    are 1/2 in square bar between the rails. Horizontal pickets are the same
    bar run across the opening. Cable styles use horizontal cable and an end
    fitting on each cable. Every post gets one piece of the selected hardware.

    Args:
        style: Railing catalog key or display name.
        length_ft: Linear footage of one run.
        height_ft: Overall height of the posts, in feet.
        mount: Hardware catalog key or display name. Defaults to a surface base plate.
        quantity: Number of matching runs.

    Returns:
        Named lines and the total weight.

    Raises:
        CatalogError: The style is unknown or the run is out of range.
    """
    chosen = lookup_railing(style)
    hardware = lookup_railing_mount(mount)
    length = _bounded_number(length_ft, "railing_length_ft", _MIN_RAILING_LENGTH_FT, _MAX_RAILING_LENGTH_FT)
    height = _bounded_number(height_ft, "railing_height_ft", _MIN_RAILING_HEIGHT_FT, _MAX_RAILING_HEIGHT_FT)
    count = _quantity(quantity)
    posts = _end_count(length, chosen.post_spacing_ft)
    lines = [
        _railing_member("posts", chosen.post, posts, height),
        _railing_member("top rail", chosen.top_rail, 1, length),
        _railing_member("bottom rail", chosen.bottom_rail, 1, length),
    ]
    if chosen.infill == "cable":
        open_in = height * 12.0 - 8.0
        cables = max(2, math.floor(open_in / RAILING_CABLE_SPACING_IN))
        lines.append(
            RailingLine(
                role="cables",
                designation="1/8 in cable",
                count=cables,
                length_ft=length,
                weight_lb=RAILING_CABLE_LB_PER_FT * length * cables,
            )
        )
        fittings = cables * 2
        lines.append(
            RailingLine(
                role="fittings",
                designation="cable end fittings",
                count=fittings,
                length_ft=0.0,
                weight_lb=RAILING_FITTING_LB * fittings,
            )
        )
    elif chosen.infill == "horizontal_picket":
        open_in = height * 12.0 - 8.0
        bars = max(2, math.floor(open_in / RAILING_HORIZONTAL_SPACING_IN))
        lines.append(
            RailingLine(
                role="bars",
                designation="1/2 in square bar",
                count=bars,
                length_ft=length,
                weight_lb=RAILING_PICKET_LB_PER_FT * length * bars,
            )
        )
    else:
        spacing = RAILING_ORNAMENTAL_SPACING_IN if chosen.infill == "ornamental" else RAILING_PICKET_SPACING_IN
        if chosen.infill == "ornamental":
            lines.append(_railing_member("mid rail", chosen.bottom_rail, 1, length))
        picket_ft = max(0.5, height - 4.0 / 12.0)
        pickets = max(0, math.floor(length * 12.0 / spacing) - posts)
        if pickets:
            lines.append(
                RailingLine(
                    role="pickets",
                    designation="1/2 in square bar",
                    count=pickets,
                    length_ft=picket_ft,
                    weight_lb=RAILING_PICKET_LB_PER_FT * picket_ft * pickets,
                )
            )
    lines.append(_railing_hardware(hardware, posts))
    takeoff = RailingTakeoff(
        style=chosen,
        mount=hardware,
        length_ft=length,
        height_ft=height,
        lines=tuple(lines),
        quantity=count,
    )
    if takeoff.weight_lb > _MAX_WEIGHT_LB:
        raise CatalogError(f"The computed weight exceeds {_MAX_WEIGHT_LB:g} lb.")
    return takeoff


def _safe_bracket(low: float, high: float) -> tuple[float, float]:
    """Round the low edge down and the high edge up, to the cent.

    The two edges stay apart by at least one cent, so the result cannot
    collapse into a single price.
    """
    if low <= 0 or high <= 0 or high < low:
        raise CatalogError("The price band could not be formed.")
    low_cents = math.floor(low * 100 + 1e-6)
    high_cents = math.ceil(high * 100 - 1e-6)
    if high_cents <= low_cents:
        high_cents = low_cents + 1
    return low_cents / 100.0, high_cents / 100.0


def _material_bracket(weight_lb: float, basis: str, summary: str, quantity: int) -> PriceBracket:
    """Price a known steel weight as a widened material bracket."""
    if weight_lb > _MAX_WEIGHT_LB:
        raise CatalogError(f"weight_lb must be {_MAX_WEIGHT_LB:g} or less.")
    raw_low = weight_lb * MATERIAL_LOW_USD_PER_LB
    raw_high = weight_lb * MATERIAL_HIGH_USD_PER_LB * (1.0 + HIGH_SIDE_ALLOWANCE)
    low, high = _safe_bracket(raw_low, raw_high)
    return PriceBracket(
        low_usd=low,
        high_usd=high,
        basis=basis,
        summary=summary,
        weight_lb=round(weight_lb, 2),
        quantity=quantity,
    )


def _plate_label(name: str) -> str:
    """Return a plate name that still reads clearly in a sentence."""
    if name.endswith("ga"):
        return name
    return f"{name} in"


def _matching_plate(thickness_in: float) -> PlateThickness | None:
    """Return the catalog plate whose thickness matches, if there is one."""
    for plate in PLATE_THICKNESSES:
        if abs(plate.thickness_in - thickness_in) <= _PLATE_MATCH_TOLERANCE_IN:
            return plate
    return None


def estimate_price(
    *,
    weight_lb: float | None = None,
    designation: str | None = None,
    length_ft: float | None = None,
    plate: str | None = None,
    thickness_in: float | None = None,
    width_in: float | None = None,
    length_in: float | None = None,
    gate_style: str | None = None,
    opening_width_ft: float | None = None,
    opening_height_ft: float | None = None,
    stair_style: str | None = None,
    step_count: int | None = None,
    stair_width_in: float | None = None,
    rise_in: float | None = None,
    run_in: float | None = None,
    patio_style: str | None = None,
    area_sq_ft: float | None = None,
    span_ft: float | None = None,
    railing_style: str | None = None,
    railing_length_ft: float | None = None,
    railing_height_ft: float | None = None,
    railing_mount: str | None = None,
    quantity: int = 1,
) -> PriceBracket:
    """Return a safe price bracket from a weight or from dimensions.

    Provide exactly one of these:

    - ``weight_lb``: price steel that has already been weighed.
    - ``designation`` and ``length_ft``: a rolled profile, HSS, or pipe.
    - ``plate`` or ``thickness_in``, plus ``width_in`` and ``length_in``.
    - ``gate_style``, ``opening_width_ft``, and ``opening_height_ft``.
    - ``stair_style``, ``step_count``, and ``stair_width_in``.
    - ``patio_style``, ``area_sq_ft``, and ``span_ft``.
    - ``railing_style``, ``railing_length_ft``, and ``railing_height_ft``, with an optional ``railing_mount``.

    Material brackets are supply prices for cut steel. The low edge uses
    ``MATERIAL_LOW_USD_PER_LB``. The high edge uses ``MATERIAL_HIGH_USD_PER_LB``
    and then ``HIGH_SIDE_ALLOWANCE`` for cuts, drops, and a short market move.
    Gate brackets use the style's area band and its minimum, and cover shop
    fabrication only. Stair brackets use the calculated flight weight and that
    style's fabrication band per pound, and cover the shop-primed flight only.
    Patio brackets name catalog profiles for the footprint and price that
    steel with the structure's fabrication band. Railing brackets name the
    posts, rails, infill, and mounting plates for the run and price that
    weight with the style's fabrication band.

    Args:
        weight_lb: Total steel weight in pounds, before quantity.
        designation: Profile or tube name from the catalog.
        length_ft: Length of one profile or tube, in feet.
        plate: Plate catalog name, such as ``1/2`` or ``10 ga``.
        thickness_in: Plate thickness in inches when it is not a catalog name.
        width_in: Plate width in inches.
        length_in: Plate length in inches.
        gate_style: Gate catalog key or display name.
        opening_width_ft: Clear opening width in feet.
        opening_height_ft: Clear opening height in feet.
        stair_style: Stair catalog key or display name.
        step_count: Number of risers in one flight.
        stair_width_in: Overall stair width in inches. For a spiral, the diameter.
        rise_in: Riser height in inches. Defaults to the catalog rise.
        run_in: Tread run in inches for a straight stair. Defaults to the catalog run.
        patio_style: Patio catalog key or display name.
        area_sq_ft: Patio footprint in square feet.
        span_ft: Primary span in feet. The other side is the area divided by this span.
        railing_style: Railing catalog key or display name.
        railing_length_ft: Linear footage of one run.
        railing_height_ft: Overall post height in feet.
        railing_mount: Hardware key or display name. Defaults to a surface base plate.
        quantity: Number of pieces, gates, flights, structures, or railing runs. ``weight_lb``,
            ``length_ft``, the plate dimensions, and the stair or patio geometry
            are for one item; quantity multiplies them.

    Returns:
        A bracket with ``low_usd`` and ``high_usd``. Average those only if a
        later step truly needs a midpoint; this function will not do it.

    Raises:
        CatalogError: The inputs mix two bases, name an unknown item, or fall
            outside the accepted size limits.
    """
    count = _quantity(quantity)
    bases = [
        name
        for name, selected in (
            ("weight", weight_lb is not None),
            ("member", designation is not None or length_ft is not None),
            ("plate", plate is not None or thickness_in is not None or width_in is not None or length_in is not None),
            ("gate", gate_style is not None or opening_width_ft is not None or opening_height_ft is not None),
            (
                "stair",
                stair_style is not None
                or step_count is not None
                or stair_width_in is not None
                or rise_in is not None
                or run_in is not None,
            ),
            ("patio", patio_style is not None or area_sq_ft is not None or span_ft is not None),
            (
                "railing",
                railing_style is not None
                or railing_length_ft is not None
                or railing_height_ft is not None
                or railing_mount is not None,
            ),
        )
        if selected
    ]
    if len(bases) != 1:
        raise CatalogError(
            "Provide one pricing basis: weight_lb; a designation with length_ft; "
            "a plate with width_in and length_in; a gate_style with opening dimensions; "
            "a stair_style with step_count and stair_width_in; "
            "a patio_style with area_sq_ft and span_ft; "
            "or a railing_style with railing_length_ft and railing_height_ft."
        )
    basis = bases[0]
    if basis == "weight":
        each = _positive_number(weight_lb, "weight_lb", _MAX_WEIGHT_LB)
        weight = each * count
        if count == 1:
            summary = f"{weight:.2f} lb of steel. Material only."
        else:
            summary = f"{each:.2f} lb each, quantity {count}, {weight:.2f} lb of steel. Material only."
        return _material_bracket(weight, "weight", summary, count)
    if basis == "member":
        if designation is None or length_ft is None:
            raise CatalogError("A profile or tube estimate needs designation and length_ft.")
        shape = _lookup_member(_require_text(designation, "designation"))
        weight = member_weight_lb(shape.designation, length_ft, count)
        length = _positive_number(length_ft, "length_ft", _MAX_LENGTH_FT)
        kind = "tube" if isinstance(shape, TubeSize) else "profile"
        summary = (
            f"{shape.designation}, {length:g} ft, quantity {count}, {weight:.2f} lb. Material only."
        )
        return _material_bracket(weight, kind, summary, count)
    if basis == "plate":
        if width_in is None or length_in is None:
            raise CatalogError("A plate estimate needs width_in and length_in.")
        if plate is not None and thickness_in is not None:
            raise CatalogError("Pass a catalog plate name or thickness_in, not both.")
        if plate is None and thickness_in is None:
            raise CatalogError("A plate estimate needs a plate name or thickness_in.")
        if plate is not None:
            chosen = lookup_plate(plate)
            thickness = chosen.thickness_in
            thickness_label = _plate_label(chosen.name)
        else:
            thickness = _positive_number(thickness_in, "thickness_in", _MAX_THICKNESS_IN)
            chosen = _matching_plate(thickness)
            thickness_label = _plate_label(chosen.name) if chosen is not None else f"{thickness:g} in"
        weight = plate_weight_lb(thickness, width_in, length_in, count)
        width = _positive_number(width_in, "width_in", _MAX_PLATE_SIDE_IN)
        length = _positive_number(length_in, "length_in", _MAX_PLATE_SIDE_IN)
        summary = (
            f"{thickness_label} plate, {width:g} in x {length:g} in, "
            f"quantity {count}, {weight:.2f} lb. Material only."
        )
        return _material_bracket(weight, "plate", summary, count)
    if basis == "gate":
        return _gate_bracket(gate_style, opening_width_ft, opening_height_ft, count)
    if basis == "stair":
        return _stair_bracket(stair_style, step_count, stair_width_in, rise_in, run_in, count)
    if basis == "patio":
        return _patio_bracket(patio_style, area_sq_ft, span_ft, count)
    return _railing_bracket(railing_style, railing_length_ft, railing_height_ft, railing_mount, count)


def _gate_bracket(
    gate_style: str | None,
    opening_width_ft: float | None,
    opening_height_ft: float | None,
    count: int,
) -> PriceBracket:
    """Price a gate from its clear opening and shop minimum."""
    if gate_style is None or opening_width_ft is None or opening_height_ft is None:
        raise CatalogError("A gate estimate needs gate_style, opening_width_ft, and opening_height_ft.")
    style = lookup_gate(gate_style)
    opening_width = _positive_number(opening_width_ft, "opening_width_ft", _MAX_OPENING_FT)
    opening_height = _positive_number(opening_height_ft, "opening_height_ft", _MAX_OPENING_FT)
    area = opening_width * opening_height
    raw_low = max(area * style.low_usd_per_sq_ft, style.minimum_low_usd) * count
    raw_high = max(area * style.high_usd_per_sq_ft, style.minimum_high_usd) * count
    low, high = _safe_bracket(raw_low, raw_high)
    summary = (
        f"{style.name}, clear opening {opening_width:g} ft x {opening_height:g} ft, "
        f"quantity {count}. Shop fabrication only."
    )
    return PriceBracket(
        low_usd=low,
        high_usd=high,
        basis="gate",
        summary=summary,
        weight_lb=None,
        quantity=count,
    )


def _stair_bracket(
    stair_style: str | None,
    step_count: int | None,
    stair_width_in: float | None,
    rise_in: float | None,
    run_in: float | None,
    count: int,
) -> PriceBracket:
    """Price a stair from its calculated steel weight and shop band."""
    if stair_style is None or step_count is None or stair_width_in is None:
        raise CatalogError("A stair estimate needs stair_style, step_count, and stair_width_in.")
    takeoff = stair_takeoff(
        stair_style,
        step_count,
        stair_width_in,
        rise_in=rise_in,
        run_in=run_in,
        quantity=count,
    )
    style = takeoff.style
    raw_low = max(takeoff.each_lb * style.low_usd_per_lb, style.minimum_low_usd) * count
    raw_high = max(takeoff.each_lb * style.high_usd_per_lb, style.minimum_high_usd) * count
    low, high = _safe_bracket(raw_low, raw_high)
    head = f"{style.name}, {takeoff.step_count} steps, {takeoff.width_in:g} in wide"
    if style.layout == "spiral":
        head = f"{head}, {takeoff.rise_in:g} in rise"
    else:
        head = f"{head}, {takeoff.rise_in:g} in rise, {takeoff.run_in:g} in run"
    pieces = [
        f"stringers {takeoff.stringer_lb:.2f} lb",
        f"treads {takeoff.tread_lb:.2f} lb",
    ]
    if takeoff.riser_lb > 0:
        pieces.append(f"risers {takeoff.riser_lb:.2f} lb")
    pieces.append(f"connectors {takeoff.connector_lb:.2f} lb")
    detail = ", ".join(pieces)
    if count == 1:
        summary = f"{head}. {detail}, {takeoff.weight_lb:.2f} lb. Shop fabrication only."
    else:
        summary = f"{head}, quantity {count}. {detail} each, {takeoff.weight_lb:.2f} lb. Shop fabrication only."
    return PriceBracket(
        low_usd=low,
        high_usd=high,
        basis="stair",
        summary=summary,
        weight_lb=round(takeoff.weight_lb, 2),
        quantity=count,
    )


def _patio_bracket(
    patio_style: str | None,
    area_sq_ft: float | None,
    span_ft: float | None,
    count: int,
) -> PriceBracket:
    """Price an outdoor frame from the profiles selected for its footprint."""
    if patio_style is None or area_sq_ft is None or span_ft is None:
        raise CatalogError("A patio estimate needs patio_style, area_sq_ft, and span_ft.")
    takeoff = patio_takeoff(patio_style, area_sq_ft, span_ft, quantity=count)
    style = takeoff.style
    raw_low = max(takeoff.each_lb * style.low_usd_per_lb, style.minimum_low_usd) * count
    raw_high = max(takeoff.each_lb * style.high_usd_per_lb, style.minimum_high_usd) * count
    low, high = _safe_bracket(raw_low, raw_high)
    members = ", ".join(
        f"{line.count} {line.role} {line.designation} at {line.length_ft:g} ft" for line in takeoff.lines
    )
    head = (
        f"{style.name}, {takeoff.area_sq_ft:g} sq ft, "
        f"{takeoff.span_ft:g} ft span by {takeoff.width_ft:g} ft wide"
    )
    if count == 1:
        summary = f"{head}. {members}, {takeoff.weight_lb:.2f} lb. Shop fabrication only."
    else:
        summary = f"{head}, quantity {count}. {members} each, {takeoff.weight_lb:.2f} lb. Shop fabrication only."
    return PriceBracket(
        low_usd=low,
        high_usd=high,
        basis="patio",
        summary=summary,
        weight_lb=round(takeoff.weight_lb, 2),
        quantity=count,
    )


def _railing_phrase(line: RailingLine) -> str:
    """Describe one railing line without a zero length."""
    if line.role in {"mounts", "fittings"}:
        return f"{line.count} {line.designation}"
    return f"{line.count} {line.role} {line.designation} at {line.length_ft:g} ft"


def _railing_bracket(
    railing_style: str | None,
    railing_length_ft: float | None,
    railing_height_ft: float | None,
    railing_mount: str | None,
    count: int,
) -> PriceBracket:
    """Price a railing from its steel, infill, selected hardware, and shop band."""
    if railing_style is None or railing_length_ft is None or railing_height_ft is None:
        raise CatalogError("A railing estimate needs railing_style, railing_length_ft, and railing_height_ft.")
    takeoff = railing_takeoff(
        railing_style,
        railing_length_ft,
        railing_height_ft,
        mount=railing_mount or RAILING_DEFAULT_MOUNT,
        quantity=count,
    )
    style = takeoff.style
    raw_low = max(takeoff.each_lb * style.low_usd_per_lb, style.minimum_low_usd) * count
    raw_high = max(takeoff.each_lb * style.high_usd_per_lb, style.minimum_high_usd) * count
    low, high = _safe_bracket(raw_low, raw_high)
    members = ", ".join(_railing_phrase(line) for line in takeoff.lines)
    head = f"{style.name}, {takeoff.length_ft:g} ft at {takeoff.height_ft:g} ft high"
    if count == 1:
        summary = f"{head}. {members}, {takeoff.weight_lb:.2f} lb. Shop fabrication only."
    else:
        summary = f"{head}, quantity {count}. {members} each, {takeoff.weight_lb:.2f} lb. Shop fabrication only."
    return PriceBracket(
        low_usd=low,
        high_usd=high,
        basis="railing",
        summary=summary,
        weight_lb=round(takeoff.weight_lb, 2),
        quantity=count,
    )


def project_family(basis: str) -> str:
    """Return the project group used to order add-ons.

    Args:
        basis: A bracket basis such as ``gate``, ``stair``, ``patio``, or ``profile``.

    Returns:
        ``gate``, ``stair``, ``patio``, ``railing``, or ``structural``.
    """
    if basis in {"gate", "stair", "patio", "railing"}:
        return basis
    return "structural"


def suggest_addons(bracket: PriceBracket) -> tuple[UpgradeOffer, ...]:
    """Return professional upgrades for a bracket, each on its own line.

    The offers follow the project type. A gate, stair, patio, or railing uses that
    type. Every other basis is treated as structural steel. The dollar
    figures are not added to the base bracket.

    Args:
        bracket: The bracket from ``estimate_price``.

    Returns:
        The upgrades for that project type, in the order an estimator should review them.
    """
    family = project_family(bracket.basis)
    by_key = {item.key: item for item in UPGRADES}
    return tuple(_price_upgrade(by_key[key], bracket, family) for key in ADDON_ORDER[family])


def _price_upgrade(upgrade: Upgrade, bracket: PriceBracket, family: str) -> UpgradeOffer:
    """Price one upgrade without folding it into the base bracket."""
    detail = upgrade.details[family]
    if upgrade.method == "added_steel":
        low, high = _safe_bracket(bracket.low_usd * upgrade.low_rate, bracket.high_usd * upgrade.high_rate)
    elif bracket.weight_lb is None:
        low, high = _safe_bracket(upgrade.minimum_low_usd, upgrade.minimum_high_usd)
        detail = f"{detail} The figures are the shop minimum until the piece is weighed."
    else:
        raw_low = max(bracket.weight_lb * upgrade.low_rate, upgrade.minimum_low_usd)
        raw_high = max(bracket.weight_lb * upgrade.high_rate, upgrade.minimum_high_usd)
        low, high = _safe_bracket(raw_low, raw_high)
    return UpgradeOffer(
        key=upgrade.key,
        name=upgrade.name,
        family=family,
        detail=detail,
        low_usd=low,
        high_usd=high,
    )


@dataclass(frozen=True)
class Recommendation:
    """One planning note that belongs with a price bracket."""

    title: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-ready dictionary."""
        return {"title": self.title, "detail": self.detail}


def recommend(
    bracket: PriceBracket,
    *,
    designation: str | None = None,
    length_ft: float | None = None,
    plate: str | None = None,
    thickness_in: float | None = None,
    width_in: float | None = None,
    length_in: float | None = None,
    gate_style: str | None = None,
    opening_width_ft: float | None = None,
    opening_height_ft: float | None = None,
    stair_style: str | None = None,
    step_count: int | None = None,
    stair_width_in: float | None = None,
    rise_in: float | None = None,
    run_in: float | None = None,
    patio_style: str | None = None,
    area_sq_ft: float | None = None,
    span_ft: float | None = None,
    railing_style: str | None = None,
    railing_length_ft: float | None = None,
    railing_height_ft: float | None = None,
) -> tuple[Recommendation, ...]:
    """Return planning notes for a bracket that has already been priced.

    The notes tell an estimator how to carry the bracket and which adjacent
    stock size, gate style, stair type, or outdoor frame to price before the
    drawing is released. They do not replace a sealed structural calculation.

    Args:
        bracket: The bracket from ``estimate_price``.
        designation: Profile or tube name, when the bracket used one.
        length_ft: Piece length in feet.
        plate: Plate catalog name.
        thickness_in: Plate thickness in inches.
        width_in: Plate width in inches.
        length_in: Plate length in inches.
        gate_style: Gate catalog key or display name.
        opening_width_ft: Clear opening width in feet.
        opening_height_ft: Clear opening height in feet.
        stair_style: Stair catalog key or display name.
        step_count: Number of risers in one flight.
        stair_width_in: Overall stair width in inches.
        rise_in: Riser height in inches.
        run_in: Tread run in inches.
        patio_style: Patio catalog key or display name.
        area_sq_ft: Patio footprint in square feet.
        span_ft: Primary span in feet.
        railing_style: Railing catalog key or display name.
        railing_length_ft: Linear footage of one run.
        railing_height_ft: Overall post height in feet.

    Returns:
        Two to four recommendations, with the bracket instruction first.
    """
    notes = [_bracket_advice(bracket), _scope_advice(bracket)]
    notes.extend(
        _specific_advice(
            bracket,
            designation=designation,
            length_ft=length_ft,
            plate=plate,
            thickness_in=thickness_in,
            width_in=width_in,
            length_in=length_in,
            gate_style=gate_style,
            opening_width_ft=opening_width_ft,
            opening_height_ft=opening_height_ft,
            stair_style=stair_style,
            step_count=step_count,
            stair_width_in=stair_width_in,
            rise_in=rise_in,
            run_in=run_in,
            patio_style=patio_style,
            area_sq_ft=area_sq_ft,
            span_ft=span_ft,
            railing_style=railing_style,
            railing_length_ft=railing_length_ft,
            railing_height_ft=railing_height_ft,
        )
    )
    return tuple(notes[:4])


def _money(value: float) -> str:
    """Format a dollar amount for a recommendation."""
    return f"${value:,.2f}"


def _bracket_advice(bracket: PriceBracket) -> Recommendation:
    """Tell the estimator to carry the low and high together."""
    spread = bracket.high_usd - bracket.low_usd
    if bracket.basis == "gate":
        room = "the room in this style's planning band for the clear opening"
    elif bracket.basis == "stair":
        room = "the room in this stair style's shop-fabrication band"
    elif bracket.basis == "patio":
        room = "the room in this outdoor frame's shop-fabrication band"
    elif bracket.basis == "railing":
        room = "the room in this railing style's shop-fabrication band"
    else:
        room = "the room between a favorable buy and a buy that includes cuts and a short market move"
    return Recommendation(
        title="Carry both edges of the bracket",
        detail=(
            f"Use {bracket.label} on the estimate. The {_money(spread)} spread is {room}. "
            "Release the purchase order from a current quote that sits inside this bracket."
        ),
    )


def _scope_advice(bracket: PriceBracket) -> Recommendation:
    """State what the bracket includes so other costs stay on their own lines."""
    if bracket.basis == "gate":
        return Recommendation(
            title="Keep the gate scope on one line",
            detail=(
                "This bracket prices a primed frame, the infill for the selected style, "
                "and standard hinges or rollers. Give galvanizing, the operator, safety loops, "
                "footings, permits, and freight their own lines."
            ),
        )
    if bracket.basis == "stair":
        return Recommendation(
            title="Keep the stair scope on one line",
            detail=(
                "This bracket prices the shop-primed flight: stringers, treads, risers when "
                "the style has them, and the connectors. Give the handrail, guard, landings, "
                "wall structure, coatings beyond primer, tax, and freight their own lines."
            ),
        )
    if bracket.basis == "patio":
        return Recommendation(
            title="Keep the outdoor frame on one line",
            detail=(
                "This bracket prices the shop-primed profiles named in the summary. "
                "Give roof panels, screen, gutters, flashing, footings, anchors, tax, "
                "and freight their own lines."
            ),
        )
    if bracket.basis == "railing":
        return Recommendation(
            title="Keep the railing scope on one line",
            detail=(
                "This bracket prices the shop-primed posts, rails, infill, and mounting plates "
                "named in the summary. Give anchors, core drilling, glass, a wood cap, coatings "
                "beyond primer, tax, and freight their own lines."
            ),
        )
    allowance = f"{HIGH_SIDE_ALLOWANCE:.0%}"
    return Recommendation(
        title="Keep this line to the steel",
        detail=(
            f"The low edge is ${MATERIAL_LOW_USD_PER_LB:.2f} per pound. "
            f"The high edge is ${MATERIAL_HIGH_USD_PER_LB:.2f} per pound plus {allowance} "
            "for drops and a short market move. Price fabrication, primer, tax, and freight separately."
        ),
    )


def _specific_advice(
    bracket: PriceBracket,
    *,
    designation: str | None,
    length_ft: float | None,
    plate: str | None,
    thickness_in: float | None,
    width_in: float | None,
    length_in: float | None,
    gate_style: str | None,
    opening_width_ft: float | None,
    opening_height_ft: float | None,
    stair_style: str | None,
    step_count: int | None,
    stair_width_in: float | None,
    rise_in: float | None,
    run_in: float | None,
    patio_style: str | None,
    area_sq_ft: float | None,
    span_ft: float | None,
    railing_style: str | None,
    railing_length_ft: float | None,
    railing_height_ft: float | None,
) -> list[Recommendation]:
    """Return the job-specific notes, most useful first."""
    notes: list[Recommendation] = []
    if bracket.basis in {"profile", "tube"} and designation and length_ft is not None:
        shape = _lookup_member(designation)
        if isinstance(shape, SteelProfile):
            notes.extend(_profile_advice(shape, length_ft))
        else:
            notes.extend(_tube_advice(shape, length_ft))
    elif bracket.basis == "plate":
        notes.extend(_plate_advice(plate, thickness_in, width_in, length_in))
    elif bracket.basis == "gate" and gate_style and opening_width_ft is not None and opening_height_ft is not None:
        notes.extend(_gate_advice(lookup_gate(gate_style), opening_width_ft, opening_height_ft))
    elif bracket.basis == "stair" and stair_style and step_count is not None and stair_width_in is not None:
        notes.extend(
            _stair_advice(
                lookup_stair(stair_style),
                step_count,
                stair_width_in,
                rise_in if rise_in is not None else STAIR_DEFAULT_RISE_IN,
                run_in if run_in is not None else STAIR_DEFAULT_RUN_IN,
                bracket,
            )
        )
    elif bracket.basis == "patio" and patio_style and area_sq_ft is not None and span_ft is not None:
        notes.extend(_patio_advice(lookup_patio(patio_style), span_ft, bracket))
    elif (
        bracket.basis == "railing"
        and railing_style
        and railing_length_ft is not None
        and railing_height_ft is not None
    ):
        notes.extend(_railing_advice(lookup_railing(railing_style), railing_length_ft, railing_height_ft, bracket))
    elif bracket.basis == "weight":
        notes.append(
            Recommendation(
                title="Name the stock size when you have it",
                detail=(
                    "A named profile, tube, or plate lets the catalog check this weight "
                    "against a stock size before the order is cut."
                ),
            )
        )
    if bracket.weight_lb is not None and bracket.weight_lb >= 2000:
        notes.append(
            Recommendation(
                title="Price the lift and the haul",
                detail=(
                    f"This takeoff is {bracket.weight_lb:,.0f} lb. Confirm the truck and the unload, "
                    "and keep that freight beside the material bracket."
                ),
            )
        )
    elif (
        bracket.weight_lb is not None
        and bracket.quantity > 0
        and bracket.weight_lb / bracket.quantity < 30
        and bracket.basis not in {"gate", "stair", "patio", "railing"}
    ):
        notes.append(
            Recommendation(
                title="Plan the short piece at the high edge",
                detail=(
                    "The setup on a small cut often costs more than the steel. "
                    "Carry the high edge until the supplier prices the drop."
                ),
            )
        )
    if bracket.quantity >= 10 and bracket.basis not in {"gate", "stair", "patio", "railing"}:
        notes.append(
            Recommendation(
                title="Send one cut list for the run",
                detail=(
                    f"The bracket multiplies all {bracket.quantity} pieces. "
                    "A single nested cut list is where a buyer finds any mill saving."
                ),
            )
        )
    return notes


def _profile_advice(profile: SteelProfile, length_ft: float) -> list[Recommendation]:
    """Recommend a stockier shape when the selected profile is light for its length."""
    notes: list[Recommendation] = []
    if length_ft < 5:
        notes.append(
            Recommendation(
                title="Confirm the cut charge",
                detail=(
                    f"{length_ft:g} ft is shorter than a typical stock length. "
                    "Ask the supplier to show the cut charge on the quote."
                ),
            )
        )
    nxt = _next_wide_flange(profile)
    if nxt is not None and length_ft >= 16:
        notes.append(
            Recommendation(
                title="Price the next heavier wide flange",
                detail=(
                    f"{profile.designation} is a light section at {length_ft:g} ft. "
                    f"If it carries floor or vehicle load, price {nxt.designation} "
                    "and check deflection before the order."
                ),
            )
        )
    thicker = _thicker_angle(profile)
    if thicker is not None:
        notes.append(
            Recommendation(
                title="Use a 1/4 in angle for a primary brace",
                detail=(
                    f"{profile.designation} suits trim and light framing. "
                    f"For a brace that shares load, price {thicker.designation}."
                ),
            )
        )
    return notes


def _tube_advice(tube: TubeSize, length_ft: float) -> list[Recommendation]:
    """Recommend a heavier wall when a thin tube is long enough to act as a post or rail."""
    notes: list[Recommendation] = []
    if length_ft < 5:
        notes.append(
            Recommendation(
                title="Confirm the cut charge",
                detail=(
                    f"{length_ft:g} ft is shorter than a typical stock length. "
                    "Ask the supplier to show the cut charge on the quote."
                ),
            )
        )
    nxt = _next_wall(tube)
    if nxt is not None and tube.wall_in <= 0.125 and length_ft >= 8:
        notes.append(
            Recommendation(
                title="Step up the wall on a long tube",
                detail=(
                    f"{tube.designation} at 1/8 in wall suits an infill frame. "
                    f"For a post or rail at {length_ft:g} ft, price {nxt.designation}."
                ),
            )
        )
    heavier = _pipe_schedule_80(tube)
    if heavier is not None and length_ft >= 8:
        notes.append(
            Recommendation(
                title="Price the Schedule 80 post",
                detail=(
                    f"When {tube.designation} is a gate post, also price {heavier.designation}. "
                    "The heavier wall is the usual choice for a tall or ornamental leaf."
                ),
            )
        )
    return notes


def _plate_advice(
    plate: str | None,
    thickness_in: float | None,
    width_in: float | None,
    length_in: float | None,
) -> list[Recommendation]:
    """Recommend a practical thickness for the plate size."""
    if plate is not None:
        chosen = lookup_plate(plate)
        thickness = chosen.thickness_in
        label = _plate_label(chosen.name)
    elif thickness_in is not None:
        thickness = thickness_in
        label = _plate_label(_matching_plate(thickness).name) if _matching_plate(thickness) else f"{thickness:g} in"
    else:
        return []
    longest = max(width_in or 0, length_in or 0)
    if thickness >= 1:
        return [
            Recommendation(
                title="Plan a burn and edge cleanup",
                detail=(
                    f"{label} plate is past a normal shear. Include burn time and edge cleanup "
                    "on a fabrication line beside this material bracket."
                ),
            )
        ]
    if thickness < 0.1875 and longest >= 36:
        return [
            Recommendation(
                title="Move a large panel up to 3/16 in",
                detail=(
                    f"A {label} panel at this size will oil-can. "
                    "Price 3/16 in or 1/4 in where the plate has to stay flat."
                ),
            )
        ]
    if thickness <= 0.1046:
        return [
            Recommendation(
                title="Treat this gauge as sheet",
                detail=(
                    f"{label} is sheet stock. Form it, or back it with a frame, "
                    "where the surface has to stay flat."
                ),
            )
        ]
    return []


def _gate_advice(style: GateStyle, width_ft: float, height_ft: float) -> list[Recommendation]:
    """Recommend a style or a scope check that matches the clear opening."""
    notes: list[Recommendation] = []
    area_low = width_ft * height_ft * style.low_usd_per_sq_ft
    if area_low < style.minimum_low_usd:
        notes.append(
            Recommendation(
                title="The shop minimum sets this bracket",
                detail=(
                    f"The clear opening is small enough that the {style.name} minimum applies. "
                    "A slightly larger opening inside that minimum stays on the same band."
                ),
            )
        )
    if style.key == "pedestrian" and (width_ft > 5 or height_ft > 8):
        notes.append(
            Recommendation(
                title="Price this opening as a vehicle leaf",
                detail=(
                    "This clear opening is past a walk gate. "
                    "Price a chain-link or ornamental swing before you keep the pedestrian style."
                ),
            )
        )
    elif style.leaf_count == 1 and style.key != "pedestrian" and "slide" not in style.key and width_ft >= 16:
        notes.append(
            Recommendation(
                title="Price a double swing or a slide",
                detail=(
                    f"A single {style.name.lower()} at {width_ft:g} ft is a long leaf. "
                    "Price a double swing or a slide gate before you release one leaf."
                ),
            )
        )
    elif style.leaf_count == 2 and width_ft >= 28:
        notes.append(
            Recommendation(
                title="Price a slide for this opening",
                detail=(
                    f"A {width_ft:g} ft clear opening is a strong fit for a cantilever or rolling slide. "
                    "Price one of those styles beside the double swing."
                ),
            )
        )
    elif style.key == "cantilever_slide" and width_ft < 12:
        notes.append(
            Recommendation(
                title="Compare a swing for a short opening",
                detail=(
                    "Under 12 ft, a swing usually hangs with less tail length. "
                    "Keep the cantilever when the drive cannot take a swing arc or a ground track."
                ),
            )
        )
    if height_ft >= 8 and style.key != "pedestrian":
        notes.append(
            Recommendation(
                title="Check wind and the operator together",
                detail=(
                    f"At {height_ft:g} ft tall, confirm the wind load and the operator's leaf rating. "
                    "The operator stays off this fabrication line."
                ),
            )
        )
    if style.leaf_count == 2:
        notes.append(
            Recommendation(
                title="The band covers both leaves",
                detail=(
                    "The width and height are the clear opening. "
                    "Both leaves, the meeting stile, and the standard hinges are inside the bracket."
                ),
            )
        )
    return notes


def _stair_advice(
    style: StairStyle,
    step_count: int,
    width_in: float,
    rise_in: float,
    run_in: float,
    bracket: PriceBracket,
) -> list[Recommendation]:
    """Recommend a scope or geometry check that matches this flight."""
    notes: list[Recommendation] = []
    each_lb = bracket.weight_lb / bracket.quantity if bracket.weight_lb is not None and bracket.quantity else 0.0
    if each_lb * style.low_usd_per_lb < style.minimum_low_usd:
        notes.append(
            Recommendation(
                title="The shop minimum sets this bracket",
                detail=(
                    f"This {style.name} flight is light enough that the shop minimum applies. "
                    "A few more steps inside that minimum stay on the same band."
                ),
            )
        )
    if style.layout == "straight" and step_count * rise_in > 144:
        notes.append(
            Recommendation(
                title="Plan a landing in this flight",
                detail=(
                    f"{step_count} steps at {rise_in:g} in is more than 12 ft of rise. "
                    "Price an intermediate landing before you release one continuous stringer."
                ),
            )
        )
    elif style.layout == "straight" and width_in < 36:
        notes.append(
            Recommendation(
                title="Confirm the walk width",
                detail=(
                    f"{width_in:g} in is under a common 36 in walk width. "
                    "Confirm the opening before this flight is detailed."
                ),
            )
        )
    elif style.key == "mono_stringer" and width_in > 48:
        notes.append(
            Recommendation(
                title="Price a channel-stringer flight beside it",
                detail=(
                    f"A {width_in:g} in mono-stringer carries the full width on one plate. "
                    "Price a channel-stringer flight before you release the center stringer."
                ),
            )
        )
    elif style.key == "floating":
        notes.append(
            Recommendation(
                title="Confirm the wall that hides the stringer",
                detail=(
                    "The cantilever arms land on structure outside this bracket. "
                    "Confirm that wall or beam before the tread size is released."
                ),
            )
        )
    elif style.layout == "spiral" and width_in < 60:
        notes.append(
            Recommendation(
                title="Confirm the walk line on this spiral",
                detail=(
                    f"A {width_in:g} in overall diameter is a tight spiral. "
                    "Confirm the clear tread at the walk line before the wedges are cut."
                ),
            )
        )
    comfort = 2 * rise_in + run_in
    if style.layout == "straight" and not 24 <= comfort <= 25:
        notes.append(
            Recommendation(
                title="Check the rise and run together",
                detail=(
                    f"Twice the {rise_in:g} in rise plus the {run_in:g} in run is {comfort:g}. "
                    "The usual comfort check lands between 24 and 25. Confirm it before the stringers are cut."
                ),
            )
        )
    return notes


def _patio_advice(style: PatioStyle, span_ft: float, bracket: PriceBracket) -> list[Recommendation]:
    """Recommend a layout check that matches this outdoor frame."""
    notes: list[Recommendation] = []
    each_lb = bracket.weight_lb / bracket.quantity if bracket.weight_lb is not None and bracket.quantity else 0.0
    if each_lb * style.low_usd_per_lb < style.minimum_low_usd:
        notes.append(
            Recommendation(
                title="The shop minimum sets this bracket",
                detail=(
                    f"This {style.name.lower()} is light enough that the shop minimum applies. "
                    "A slightly larger footprint inside that minimum stays on the same band."
                ),
            )
        )
    if style.kind != "railing" and span_ft > 20:
        notes.append(
            Recommendation(
                title="Price this span as two bays",
                detail=(
                    f"A single {span_ft:g} ft span is at the long end of this planning table. "
                    "Price a center support or a second bay before one rafter is released."
                ),
            )
        )
    elif style.kind != "railing" and span_ft > 16:
        notes.append(
            Recommendation(
                title="Confirm the center beam line",
                detail=(
                    "A third post line is in this weight because the span is over 16 ft. "
                    "Confirm that line on the plan before the rafters are cut."
                ),
            )
        )
    elif style.kind != "railing":
        notes.append(
            Recommendation(
                title="A ledger can replace one post line",
                detail=(
                    "The weight includes two post lines. After the wall is checked, "
                    "a ledger can replace the line at the house."
                ),
            )
        )
    else:
        notes.append(
            Recommendation(
                title="Confirm what the posts land on",
                detail=(
                    "These posts are planned at 6 ft around the footprint. "
                    "Confirm the slab, deck, or curb, and keep the anchors off this line."
                ),
            )
        )
    return notes


def _railing_advice(
    style: RailingStyle,
    length_ft: float,
    height_ft: float,
    bracket: PriceBracket,
) -> list[Recommendation]:
    """Recommend a height or splice check that matches this railing."""
    notes: list[Recommendation] = []
    each_lb = bracket.weight_lb / bracket.quantity if bracket.weight_lb is not None and bracket.quantity else 0.0
    if each_lb * style.low_usd_per_lb < style.minimum_low_usd:
        notes.append(
            Recommendation(
                title="The shop minimum sets this bracket",
                detail=(
                    f"This {style.name.lower()} is light enough that the shop minimum applies. "
                    "A slightly longer run inside that minimum stays on the same band."
                ),
            )
        )
    if style.kind in {"exterior", "cable", "balcony"} and height_ft < 3.5:
        notes.append(
            Recommendation(
                title="Confirm the guard height",
                detail=(
                    f"This run is {height_ft:g} ft high. A guard is usually 42 in. "
                    "Confirm the height before the posts are cut."
                ),
            )
        )
    elif style.kind == "interior" and height_ft > 3.5:
        notes.append(
            Recommendation(
                title="Confirm the opening this rail guards",
                detail=(
                    f"{height_ft:g} ft is taller than a typical 36 in interior rail. "
                    "Confirm whether this opening needs a guard height."
                ),
            )
        )
    elif style.infill == "cable":
        notes.append(
            Recommendation(
                title="Keep glass and a wood cap off this line",
                detail=(
                    "The cables and end fittings are already in this weight. "
                    "A glass panel or a wood cap stays on its own line."
                ),
            )
        )
    if length_ft > 20:
        notes.append(
            Recommendation(
                title="Plan the splice or the corner",
                detail=(
                    f"A {length_ft:g} ft run needs a rail splice or a corner post where it turns. "
                    "Confirm that joint on the layout before the rail is cut."
                ),
            )
        )
    return notes


def _next_wide_flange(profile: SteelProfile) -> SteelProfile | None:
    """Return the next heavier wide flange of the same nominal depth."""
    if profile.family != "wide_flange" or profile.weight_lb_per_ft > 14:
        return None
    prefix = profile.designation.split("x", 1)[0] + "x"
    heavier = [
        item
        for item in STEEL_PROFILES
        if item.family == "wide_flange"
        and item.designation.startswith(prefix)
        and item.weight_lb_per_ft > profile.weight_lb_per_ft
    ]
    if not heavier:
        return None
    return min(heavier, key=lambda item: item.weight_lb_per_ft)


def _thicker_angle(profile: SteelProfile) -> SteelProfile | None:
    """Return the 1/4 in angle with the same legs, when the selected angle is lighter."""
    if profile.family != "angle":
        return None
    stem, thickness = profile.designation.rsplit("x", 1)
    if thickness not in {"1/8", "3/16"}:
        return None
    target = f"{stem}x1/4"
    for item in STEEL_PROFILES:
        if item.designation == target:
            return item
    return None


def _next_wall(tube: TubeSize) -> TubeSize | None:
    """Return the next thicker HSS of the same outside size."""
    if tube.shape == "pipe":
        return None
    candidates = [
        item
        for item in TUBE_SIZES
        if item.shape == tube.shape
        and item.outside_width_in == tube.outside_width_in
        and item.outside_height_in == tube.outside_height_in
        and item.wall_in > tube.wall_in + 1e-6
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item.wall_in)


def _pipe_schedule_80(tube: TubeSize) -> TubeSize | None:
    """Return Schedule 80 pipe of the same nominal size."""
    if tube.shape != "pipe" or tube.schedule != "40":
        return None
    for item in TUBE_SIZES:
        if item.shape == "pipe" and item.nominal_size == tube.nominal_size and item.schedule == "80":
            return item
    return None


__all__ = [
    "HIGH_SIDE_ALLOWANCE",
    "MATERIAL_HIGH_USD_PER_LB",
    "MATERIAL_LOW_USD_PER_LB",
    "PLATE_THICKNESSES",
    "GATE_STYLES",
    "STAIR_DEFAULT_RISE_IN",
    "STAIR_DEFAULT_RUN_IN",
    "STAIR_SPIRAL_DEGREES_PER_STEP",
    "STAIR_SPIRAL_INNER_RADIUS_IN",
    "STAIR_STYLES",
    "PATIO_POST_HEIGHT_FT",
    "PATIO_POST_SPACING_FT",
    "PATIO_RAILING_HEIGHT_FT",
    "PATIO_RAILING_POST_SPACING_FT",
    "PATIO_STYLES",
    "RAILING_CABLE_LB_PER_FT",
    "RAILING_CABLE_SPACING_IN",
    "RAILING_FITTING_LB",
    "RAILING_DEFAULT_MOUNT",
    "RAILING_HORIZONTAL_SPACING_IN",
    "RAILING_MOUNTS",
    "RAILING_ORNAMENTAL_SPACING_IN",
    "RAILING_PICKET_LB_PER_FT",
    "RAILING_PICKET_SPACING_IN",
    "RAILING_STYLES",
    "STEEL_DENSITY_LB_PER_CU_FT",
    "STEEL_LB_PER_SQ_FT_PER_INCH",
    "STEEL_PROFILES",
    "STRUCTURAL_STEEL_WEIGHTS",
    "TUBE_SIZES",
    "ADDON_ORDER",
    "UPGRADES",
    "CatalogError",
    "GateStyle",
    "PatioLine",
    "PatioStyle",
    "PatioTakeoff",
    "PlateThickness",
    "PriceBracket",
    "RailingLine",
    "RailingMount",
    "RailingStyle",
    "RailingTakeoff",
    "Recommendation",
    "Upgrade",
    "UpgradeOffer",
    "StairStyle",
    "StairTakeoff",
    "SteelProfile",
    "TubeSize",
    "estimate_price",
    "recommend",
    "lookup_gate",
    "lookup_patio",
    "lookup_plate",
    "lookup_profile",
    "lookup_railing",
    "lookup_railing_mount",
    "lookup_stair",
    "lookup_tube",
    "member_weight_lb",
    "patio_takeoff",
    "plate_weight_lb",
    "railing_takeoff",
    "project_family",
    "stair_takeoff",
    "suggest_addons",
]