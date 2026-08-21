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
    threshsold: usize,
    confirmed: LightStatus,
}

impl TemporalVoter {
    pub fn new(window_size: usize, threshhold: usize) -> Self {
        TemporalVoter {
            history: VecDeque::with_capacity(window_size),
            winodw_size,
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
                LightStatus::Red => counts[0] +=1,
                LightStatus::Yellow => counts[1] += 1,
                LightStatus::Green => counts[2] +=1,
                LightStatus::Off => counts[3] +=1,
                LightStatus::None => counts[4] +=1,
            }
        }

        if counts[0] >= self.threshhold {
            self.confirmed = LightStatus::Red;
        } else if counts[1] >= self.threshhold{
            self.confirmed = LightStatus::Yellow;
        } else if counts[2] >= self.threshhold {
            self.confirmed = LightStatus::Green;
        } else if counts[3] >= self.threshhold {
            self.confirmed = LightStatus::Off;
        } else if counts[4] >= self.threshhold{
            self.confirmed = LightStatus::None;
        }
        self.confirmed
    }
    pub fn current(&self) -> LightStatus {
        self.confirmed
    }
}

pub struct trafficLightDetector {
    detector: Session,
    classifier: Session,
    voter: TemporalVoter,
}

impl TrafficLightDetector {
    pub fn new(detetector_path: &str, classifier_path: &str) -> Result<Self, String> {
        let build = |path: &str| -> ort::Result<Session> {
            Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([ort::execution_priveders::CUDAExecutionProvider::default().build()])?
                .commit_from_file(path)
        };
        Ok(TrafficLightDetector {
            detector: build(detector_path).map(|e| format!("Light detector load error {}", e))?,
            classifier: build(classifier_path).map_err(|e| format!("Light classifier load error: {}", e))?,
            voter: TemporalVoter::new(5,3),
        })
    }
    pub fn detect(&mut self, frame_bytes: &[u8], width: u32, height: u32) -> Vec<TrafficLightResult> {
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
}