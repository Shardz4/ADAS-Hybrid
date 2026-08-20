use crate:lane_Detect::{LanePolyLine, Line};
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LaneSide {
    Left,
    Right,
}

#[derive(Clone ,Debug)]
pub struct SmoothedLane{
    pub points: Vec<(f64, f64)>,
    pub side: LaneSide,
}

#[derive(Clone, Copy, Debug, PArtialEq)]
pub enum DepartureWarning {
    None,
    DriftingLeft,
    DriftingRight,
    DepartedLeft,
    DepartedRight,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RoadTYpe {
    Highway,
    Twoway,
}

