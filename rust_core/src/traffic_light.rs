use ndarray::ArrayView3;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightStatus {
    Red,
    Yellow,
    Green,
    Off,
    None,
}

#[derive(Debug, Clone)]
pub struct TrafficLightResult {
    pub bbox: [f32; 4],
    pub status: LightStatus,
    pub det_confidence: f32,
    pub cls_confidence: f32,
    pub voted_status: LightStatus,
}

pub struct TemporalVoter {
    history: VecDeque<LightStatus>,
    window_size: usize,
    threshold: usize,
    confirmed: LightStatus,
}

impl TemporalVoter {
    pub fn new(window_size: usize, threshold: usize) -> Self {
        TemporalVoter {
            history: VecDeque::with_capacity(window_size),
            window_size,
            threshold,
            confirmed: LightStatus::None,
        }
    }

    pub fn vote(&mut self, observed: LightStatus) -> LightStatus {
        self.history.push_back(observed);
        if self.history.len() > self.window_size {
            self.history.pop_front();
        }

        let mut counts = [0usize; 5];
        for s in &self.history {
            match s {
                LightStatus::Red => counts[0] += 1,
                LightStatus::Yellow => counts[1] += 1,
                LightStatus::Green => counts[2] += 1,
                LightStatus::Off => counts[3] += 1,
                LightStatus::None => counts[4] += 1,
            }
        }

        if counts[0] >= self.threshold {
            self.confirmed = LightStatus::Red;
        } else if counts[1] >= self.threshold {
            self.confirmed = LightStatus::Yellow;
        } else if counts[2] >= self.threshold {
            self.confirmed = LightStatus::Green;
        } else if counts[3] >= self.threshold {
            self.confirmed = LightStatus::Off;
        } else if counts[4] >= self.threshold {
            self.confirmed = LightStatus::None;
        }
        self.confirmed
    }

    pub fn current(&self) -> LightStatus {
        self.confirmed
    }
}

pub struct TrafficLightDetector {
    detector: Session,
    classifier: Session,
    voter: TemporalVoter,
}

impl TrafficLightDetector {
    pub fn new(detector_path: &str, classifier_path: &str) -> Result<Self, String> {
        let build = |path: &str| -> ort::Result<Session> {
            Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_file(path)
        };

        Ok(TrafficLightDetector {
            detector: build(detector_path)
                .map_err(|e| format!("Light detector load error: {}", e))?,
            classifier: build(classifier_path)
                .map_err(|e| format!("Light classifier load error: {}", e))?,
            voter: TemporalVoter::new(5, 3),
        })
    }

    pub fn detect(
        &mut self,
        frame_bytes: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<TrafficLightResult> {
        let fixture_bboxes = self.detect_fixtures(frame_bytes, width, height);
        if fixture_bboxes.is_empty() {
            self.voter.vote(LightStatus::None);
            return vec![];
        }

        let mut results = Vec::new();
        for (bbox, det_conf) in &fixture_bboxes {
            let crop = self.crop_and_resize(frame_bytes, width, height, bbox, 64, 32);
            let (status, cls_conf) = self.classify_crop(&crop);
            let voted = self.voter.vote(status);

            results.push(TrafficLightResult {
                bbox: *bbox,
                status,
                det_confidence: *det_conf,
                cls_confidence: cls_conf,
                voted_status: voted,
            });
        }
        results
    }

    fn detect_fixtures(&mut self, frame_bytes: &[u8], w: u32, h: u32) -> Vec<([f32; 4], f32)> {
        let roi_h = (h as f32 * 0.6) as u32;
        let mut chw = vec![0.0f32; 3 * 320 * 320];

        for y in 0..320usize {
            let src_y = ((y as f32 / 320.0) * roi_h as f32) as u32;
            for x in 0..320usize {
                let src_x = ((x as f32 / 320.0) * w as f32) as u32;
                let src_idx = ((src_y * w + src_x) * 3) as usize;
                if src_idx + 2 < frame_bytes.len() {
                    let b = frame_bytes[src_idx] as f32 / 255.0;
                    let g = frame_bytes[src_idx + 1] as f32 / 255.0;
                    let r = frame_bytes[src_idx + 2] as f32 / 255.0;

                    let idx = y * 320 + x;
                    chw[idx] = r;
                    chw[320 * 320 + idx] = g;
                    chw[2 * 320 * 320 + idx] = b;
                }
            }
        }

        let tensor = match Tensor::from_array(([1usize, 3, 320, 320], chw)) {
            Ok(t) => t,
            Err(_) => return vec![],
        };

        let outputs = match self.detector.run(ort::inputs!["images" => tensor]) {
            Ok(o) => o,
            Err(_) => return vec![],
        };

        let (_, data) = match outputs[0].try_extract_tensor::<f32>() {
            Ok(d) => d,
            Err(_) => return vec![],
        };

        // YOLO parse: shape [1, 5, 2100] -> cx, cy, w_box, h_box, conf
        let num_anchors = 2100;
        let mut candidates = Vec::new();

        for i in 0..num_anchors {
            let conf = data[4 * num_anchors + i];
            if conf > 0.35 {
                let cx = data[0 * num_anchors + i] * (w as f32 / 320.0);
                let cy = data[1 * num_anchors + i] * (roi_h as f32 / 320.0);
                let bw = data[2 * num_anchors + i] * (w as f32 / 320.0);
                let bh = data[3 * num_anchors + i] * (roi_h as f32 / 320.0);

                let x1 = (cx - bw / 2.0).max(0.0);
                let y1 = (cy - bh / 2.0).max(0.0);
                let x2 = (cx + bw / 2.0).min(w as f32);
                let y2 = (cy + bh / 2.0).min(h as f32);

                candidates.push(([x1, y1, x2, y2], conf));
            }
        }

        // Greedy NMS
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut kept: Vec<([f32; 4], f32)> = Vec::new();
        for cand in candidates {
            let mut overlap = false;
            for k in &kept {
                if bbox_iou(&cand.0, &k.0) > 0.45 {
                    overlap = true;
                    break;
                }
            }
            if !overlap {
                kept.push(cand);
            }
        }
        kept
    }

    fn crop_and_resize(
        &self,
        frame_bytes: &[u8],
        w: u32,
        _h: u32,
        bbox: &[f32; 4],
        target_h: u32,
        target_w: u32,
    ) -> Vec<f32> {
        let x1 = bbox[0].max(0.0) as u32;
        let y1 = bbox[1].max(0.0) as u32;
        let x2 = bbox[2].max(0.0) as u32;
        let y2 = bbox[3].max(0.0) as u32;

        let crop_w = (x2.saturating_sub(x1)).max(1);
        let crop_h = (y2.saturating_sub(y1)).max(1);

        let mut chw = vec![0.0f32; (3 * target_h * target_w) as usize];

        for y in 0..target_h {
            let src_y = y1 + (y as f32 / target_h as f32 * crop_h as f32) as u32;
            for x in 0..target_w {
                let src_x = x1 + (x as f32 / target_w as f32 * crop_w as f32) as u32;
                let idx = ((src_y * w + src_x) * 3) as usize;

                if idx + 2 < frame_bytes.len() {
                    let b = frame_bytes[idx] as f32 / 255.0;
                    let g = frame_bytes[idx + 1] as f32 / 255.0;
                    let r = frame_bytes[idx + 2] as f32 / 255.0;

                    let out_idx = (y * target_w + x) as usize;
                    chw[out_idx] = r;
                    chw[(target_h * target_w) as usize + out_idx] = g;
                    chw[(2 * target_h * target_w) as usize + out_idx] = b;
                }
            }
        }
        chw
    }

    fn classify_crop(&mut self, crop_chw: &[f32]) -> (LightStatus, f32) {
        let tensor = match Tensor::from_array(([1usize, 3, 64, 32], crop_chw.to_vec())) {
            Ok(t) => t,
            Err(_) => return (LightStatus::None, 0.0),
        };

        let outputs = match self.classifier.run(ort::inputs!["input" => tensor]) {
            Ok(o) => o,
            Err(_) => return (LightStatus::None, 0.0),
        };

        let (_, logits) = match outputs[0].try_extract_tensor::<f32>() {
            Ok(d) => d,
            Err(_) => return (LightStatus::None, 0.0),
        };

        // Softmax over 4 classes: Red, Yellow, Green, Off
        let mut max_val = f32::NEG_INFINITY;
        let mut sum = 0.0f32;
        let mut exp_vals = [0.0f32; 4];

        for (i, &l) in logits.iter().take(4).enumerate() {
            let e = l.exp();
            exp_vals[i] = e;
            sum += e;
        }

        let mut max_idx = 0;
        for (i, e) in exp_vals.iter_mut().enumerate() {
            *e /= sum;
            if *e > max_val {
                max_val = *e;
                max_idx = i;
            }
        }

        let status = match max_idx {
            0 => LightStatus::Red,
            1 => LightStatus::Yellow,
            2 => LightStatus::Green,
            _ => LightStatus::Off,
        };
        (status, max_val)
    }
}

fn bbox_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter = inter_w * inter_h;

    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union = area_a + area_b - inter;

    if union > 0.0 { inter / union } else { 0.0 }
}

pub fn detect_traffic_light_hsv(frame: &ArrayView3<u8>) -> LightStatus {
    let (h, w) = (frame.dim().0, frame.dim().1);
    let scan_h = h / 3;

    let mut red_count = 0u32;
    let mut green_count = 0u32;
    let mut yellow_count = 0u32;

    for y in 0..scan_h {
        for x in 0..w {
            let b = frame[[y, x, 0]] as f32;
            let g = frame[[y, x, 1]] as f32;
            let r = frame[[y, x, 2]] as f32;

            if r > 180.0 && g < 80.0 && b < 80.0 {
                red_count += 1;
            } else if g > 180.0 && r < 80.0 && b < 80.0 {
                green_count += 1;
            } else if r > 180.0 && g > 180.0 && b < 80.0 {
                yellow_count += 1;
            }
        }
    }

    if red_count > 50 {
        LightStatus::Red
    } else if yellow_count > 50 {
        LightStatus::Yellow
    } else if green_count > 50 {
        LightStatus::Green
    } else {
        LightStatus::None
    }
}