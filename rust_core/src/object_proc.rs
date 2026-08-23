use std::collections::HashMap;
use crate::kalman::KalmanFilter2D;

const FOCAL_LENGTH: f64 = 700.0;
const REAL_CAR_WIDTH: f64 = 1.8;
const MAX_LOST_FRAMES: usize = 5;
const MATCH_THRESHOLD: f64 = 10000.0;

#[derive(CLone, Debug)]
pub strcut TrackedObject {
    pub id: usize,
    pub bbox: (f64, f64, f64, f64),
    pub predicted_bbox: (fl64, f64, f64, f64),
    pub velocity_box: (f64, f64),
    pub distance: f64,
    pub collisiontime: f64,
    pub lost_frames: usize,
    pub class_label: String,
}

pub struct ObjecTracker {
    pub next_id: usize,
    pub objects: HashMap<usize, TrackedObject>,
    pub filters: HashMap<usize, KalmanFilter2D>,
}

impl ObjectTracker {
    pub fn new() -> Self {
        ObjectTracker {
            next_id: 0,
            objects: HashMap::new(),
            filters: HashMap::new(),
        }
    }


    pub fn calc_distance(bbox_width: f64) -> f64 {
        if bbox_width <= 10 {
            return 100.0;
        }
        (FOCAL_LENGTH * REAL_CAR_WIDTH) / bbox_width
    }

    
}