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

