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