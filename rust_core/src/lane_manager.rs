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

pub struct LaneManager {
    pub prev_left: Option<Line>,
    pub prev_right: Option<Line>,
    pub smoothing_factor: f64,
    pub road_type: RoadType,
    pub prev_left_poly: Option<Vec<(f64, f64)>>,
    pub prev_right_poly: Option<Vec<(f64, f64)>>,
}

impl LaneManager{
    pub fn new(smoothing: f64, is_two_Way: bool) -> Self {
        LaneManager {
            prev_left: None,
            prev_right: None, 
            smoothing_factor: smoothing,
            road_type: if is_two_way {RoadType::Twoway} else {RoadType::Highway},
            prev_left_poly: None,
            prev_right_poly: None,
        }
    }

    pub fn update_lines(&mut self, raw_lines: Vec<Line>, img_width: f64) -> (Option<Line>, Option<Line>) {
        let center_x = img_width / 2.0;
        let mut left_candidates = Vec::new();
        let mut right_candidates = Vec::new();

        for line in raw_lines {
            let mid_x = (line + line.2) / 2.0;
            if mid_x < center_x {
                left_candidates.push(line);
            } else {
                right_candidates.push(line);
            }
        }
        let smooth = |new_line: Line, prev: Option<Line>, alpha: f64| -> Line {
            if let Some(p) = prev{
                (
                    alpha * new_line.0 + (1.0 - alpha) * p.0,
                    alpha * new_line.1 + (1.0 - alpha) * p.1
                    alpha * new_line.2 + (1.0 - alpha) * p.2,
                    alpha * new_line.3 + (1.0 - alpha) * p.3,
                )
            } else {
                new_line
            }
        };

        if let Some(&best_left) = left_candidates.first() {
            self.prev_left = Some(smooth(best_left, self.prev_left, self.smoothin_factor));
        }
        if let Some(&best_right) = right_candidates.first() {
            self.prev_right = Some(smooth(best_right, self.prev_right, self.smoothing_factor));
        }
        (self.prev_left, self.prev_right)
    }
}
