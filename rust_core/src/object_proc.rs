use std::collections::HashMap;
use crate::kalman::KalmanFilter2D;

const FOCAL_LENGTH: f64 = 700.0;
const REAL_CAR_WIDTH: f64 = 1.8;
const MAX_LOST_FRAMES: usize = 5;
const MATCH_THRESHOLD: f64 = 10000.0;

#[derive(Clone, Debug)]
pub struct TrackedObject {
    pub id: usize,
    pub bbox: (f64, f64, f64, f64),
    pub predicted_bbox: (f64, f64, f64, f64),
    pub velocity: (f64, f64),
    pub distance: f64,
    pub speed: f64,
    pub collisiontime: f64,
    pub lost_frames: usize,
    pub class_label: String,
}

pub struct ObjectTracker {
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
        if bbox_width <= 1.0 {
            return 100.0;
        }
        (FOCAL_LENGTH * REAL_CAR_WIDTH) / bbox_width
    }

    pub fn process_frame(
        &mut self,
        detections: Vec<(f64, f64, f64, f64, String)>,
        dt: f64,
    ) -> Vec<TrackedObject> {
        // Step 1: Advance Kalman filters and update predicted bounding boxes
        for (id, filter) in &mut self.filters {
            let predicted = filter.predict(dt);
            if let Some(obj) = self.objects.get_mut(id) {
                obj.predicted_bbox = (
                    predicted[0] - obj.bbox.2 / 2.0,
                    predicted[1] - obj.bbox.3 / 2.0,
                    obj.bbox.2,
                    obj.bbox.3,
                );
            }
        }

        // Step 2: Associate detections to existing tracks
        let mut new_objects = HashMap::new();
        let mut new_filters = HashMap::new();
        let mut matched_ids = Vec::new();

        for (x, y, w, h, label) in detections {
            let cx = x + w / 2.0;
            let cy = y + h / 2.0;
            let current_dist = Self::calc_distance(w);

            let mut best_match = None;
            let mut min_error = f64::MAX;

            for (id, obj) in &self.objects {
                if matched_ids.contains(id) {
                    continue;
                }

                let (pcx, pcy) = if let Some(filter) = self.filters.get(id) {
                    let st = filter.state();
                    (st.0, st.1)
                } else {
                    let (ox, oy, ow, oh) = obj.bbox;
                    (ox + ow / 2.0, oy + oh / 2.0)
                };

                let error = (cx - pcx).powi(2) + (cy - pcy).powi(2);
                if error < MATCH_THRESHOLD && error < min_error {
                    min_error = error;
                    best_match = Some(*id);
                }
            }

            if let Some(id) = best_match {
                matched_ids.push(id);

                if let Some(mut filter) = self.filters.remove(&id) {
                    filter.update([cx, cy]);
                    let state = filter.state();
                    new_filters.insert(id, filter);

                    let prev_dist = self.objects.get(&id).map(|o| o.distance).unwrap_or(current_dist);
                    let speed = (prev_dist - current_dist) / dt;
                    let ttc = if speed > 0.1 { current_dist / speed } else { 99.0 };

                    new_objects.insert(
                        id,
                        TrackedObject {
                            id,
                            bbox: (x, y, w, h),
                            predicted_bbox: (state.0 - w / 2.0, state.1 - h / 2.0, w, h),
                            velocity: (state.2, state.3),
                            distance: current_dist,
                            speed,
                            collisiontime: ttc,
                            lost_frames: 0,
                            class_label: label,
                        },
                    );
                }
            } else {
                // Register new track
                let id = self.next_id;
                self.next_id += 1;

                let filter = KalmanFilter2D::new(cx, cy);
                new_filters.insert(id, filter);

                new_objects.insert(
                    id,
                    TrackedObject {
                        id,
                        bbox: (x, y, w, h),
                        predicted_bbox: (x, y, w, h),
                        velocity: (0.0, 0.0),
                        distance: current_dist,
                        speed: 0.0,
                        collisiontime: 99.0,
                        lost_frames: 0,
                        class_label: label,
                    },
                );
            }
        }

        // Step 3: Retain ghost tracks for occlusions
        for (id, obj) in &self.objects {
            if !matched_ids.contains(id) && obj.lost_frames < MAX_LOST_FRAMES {
                let mut ghost = obj.clone();
                ghost.lost_frames += 1;
                ghost.bbox = ghost.predicted_bbox;

                new_objects.insert(*id, ghost);
                if let Some(f) = self.filters.remove(id) {
                    new_filters.insert(*id, f);
                }
            }
        }

        self.objects = new_objects;
        self.filters = new_filters;
        self.objects.values().cloned().collect()
    }
}