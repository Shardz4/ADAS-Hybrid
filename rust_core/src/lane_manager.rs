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

    pub fn update_polylines(&mut self, lanes: Vec<LanePolyLine>, img_width: f64,) -> (Option<SmoothedLane>, Option<SmoothedLane>) {
        let center_x = img_width / 2.0;
        let mut left_poly: Option<Vec<(f64, f64)>> = None;
        let mut right_poly: Option<Vec<(f64, f64)>> = None;

        for lane in lanes {
            if let Some(bottom_pt) = lane.points.first() {
                if bottom_pt.0 < center_x {
                    if left_poly.is_none() {
                        left_poly = Some(lane.points.clone());
                    }
                } else if right_poly.is_none(){
                    right_poly = Some(lane_points.clone());
                }
            }
        }

        let smooth_poly = |new_p: Vec<(f64, f64)>, prev_p: Option<Vec<(f64, f64)>>, alpha: f64| -> Vec<(f64, f64)> {
            if let Some(prev) = prev_p {
                new_p.iter().zip(prev.iter()).map(|(new_pt, prev_pt)| {
                    (
                        alpha * new_pt.0 + (1.0 - alpha) * prev_pt.0,
                        alpha * new_pt.1 + (1.0 - alpha) * prev_pt.1,
                    )
                }).collect()
            } else {
                new_p
            }
        };

        let mut res_left = None;
        let mut res_right = None;

        if let Some(lp) = left_poly {
            let smoothed = smooth_poly(lp, self.prev_left_poly.clone(), self.smoothing_factor);
            if let (Some(first), Some(last)) = (smoothed.first(), smoothed.last()) {
                self.prev_left = Some((first.0, first.1, last.0, last.1));;
            }
            self.prev_left_poly = Some(smoothed.clone());
            res_left = Some(SmoothedLane{ points: smoothed, side: LaneSide::Left});
        }

        if let Some(rp) = right_poly {
            let smoothed = smooth_poly(rp, self.prev_right_poly.clone(), self.smoothing_factor);
            if let (Some(first), Some(last)) = (smoothed.first(), smoothed.last()) {
                self.prev_right = Some((first.0, first.1, last.0, last.1));
            }
            self.prev_right_poly = Some(smoothed.clone());
            res_right = Some(SmoothedLane{ points: smoothed, side: LaneSide::Right});
        }
        (res_left, res_right)
    }

    pub fn check_departure(&self, img_width: f64, _img_height: f64) -> DepartureWarning {
        let left_x = self.prev_left_poly.as_ref().and_then(|p| p.first().map(|pt| pt.0)).or_else(|| self.prev_left.map(|l| l.0));
        let right_x = self.prev_right_poly.as_ref().and_then(|p| p.first().map(|pt| pt.0)).or_else(|| self.prev_right.map(|l| l.0));
        
        if let (Some(lx), Some(rx)) = (left_x, right_x) {
            let lane_Width = rx - lx;
            if lane_Width <= 0.0 {
                return DepartureWarning::None;
            }
            let img_center = img_width / 2.0;
            let lane_center = lx + lane_Width / 2.0;
            let offset = img_center - lane_center;

            if offset > lane_Width * 0.50 {
                DepartureWarning::DepartedRight
            } else if offset < -lane_Width * 0.50 {
                DepartureWarning::DepartedLeft
            } else if offset > lane_Width * 0.35{
                DepartureWarning::DriftingRight
            } else if offset < -lane_Width * 0.35 {
                DepartureWarning::DriftingLeft
            } else {
                DepartureWarning::None
            }
        } else {
            DepartureWarning::None
        }
    }
}
