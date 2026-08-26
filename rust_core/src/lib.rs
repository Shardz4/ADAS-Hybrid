pub mod kalman;
pub mod lane_detect;
pub mod lane_manager;
pub mod object_proc;
pub mod traffic_light;

use numpy::PyReadonlyArray3;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use lane_manager::LaneManager as CoreLaneManager;
use object_proc::ObjectTracker;
use traffic_light::{detect_traffic_light_hsv, LightStatus, TrafficLightDetector};

#[pyclass]
pub struct AdasBrain {
    vehicle_session: Session,
    sign_session: Session,
    lane_session: Option<Session>,
    light_detector: Option<TrafficLightDetector>,
}

#[pymethods]
impl AdasBrain {
    #[new]
    #[pyo3(signature = (vehicle_model, sign_model, lane_model=None, light_det_model=None, light_cls_model=None))]
    pub fn new(
        vehicle_model: &str,
        sign_model: &str,
        lane_model: Option<&str>,
        light_det_model: Option<&str>,
        light_cls_model: Option<&str>,
    ) -> PyResult<Self> {
        let build = |path: &str| -> Result<Session, String> {
            Session::builder()
                .map_err(|e| e.to_string())?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| e.to_string())?
                .with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default().build()])
                .map_err(|e| e.to_string())?
                .commit_from_file(path)
                .map_err(|e| e.to_string())
        };

        let vehicle_session = build(vehicle_model).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        let sign_session = build(sign_model).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        let lane_session = match lane_model {
            Some(path) => Some(build(path).map_err(pyo3::exceptions::PyRuntimeError::new_err)?),
            None => None,
        };

        let light_detector = match (light_det_model, light_cls_model) {
            (Some(det), Some(cls)) => {
                Some(TrafficLightDetector::new(det, cls).map_err(pyo3::exceptions::PyRuntimeError::new_err)?)
            }
            _ => None,
        };

        Ok(AdasBrain {
            vehicle_session,
            sign_session,
            lane_session,
            light_detector,
        })
    }

    pub fn detect_vehicles(
        &self,
        py: Python,
        frame_bytes: &[u8],
        width: u32,
        height: u32,
        conf_threshold: f32,
    ) -> PyResult<PyObject> {
        let detections = py.allow_threads(|| {
            let mut chw = vec![0.0f32; 3 * 640 * 640];
            let x_scale = width as f32 / 640.0;
            let y_scale = height as f32 / 640.0;

            for y in 0..640 {
                let src_y = ((y as f32 * y_scale) as u32).min(height - 1);
                for x in 0..640 {
                    let src_x = ((x as f32 * x_scale) as u32).min(width - 1);
                    let src_idx = ((src_y * width + src_x) * 3) as usize;
                    if src_idx + 2 < frame_bytes.len() {
                        let b = frame_bytes[src_idx] as f32 / 255.0;
                        let g = frame_bytes[src_idx + 1] as f32 / 255.0;
                        let r = frame_bytes[src_idx + 2] as f32 / 255.0;

                        let idx = y * 640 + x;
                        chw[idx] = r;
                        chw[640 * 640 + idx] = g;
                        chw[2 * 640 * 640 + idx] = b;
                    }
                }
            }

            let tensor = match Tensor::from_array(([1usize, 3, 640, 640], chw)) {
                Ok(t) => t,
                Err(_) => return Vec::new(),
            };

            let outputs = match self.vehicle_session.run(ort::inputs!["images" => tensor].unwrap()) {
                Ok(o) => o,
                Err(_) => return Vec::new(),
            };

            let (_, data) = match outputs[0].try_extract_tensor::<f32>() {
                Ok(d) => d,
                Err(_) => return Vec::new(),
            };

            let num_anchors = 8400;
            let mut boxes = Vec::new();

            for i in 0..num_anchors {
                let mut max_cls_conf = 0.0f32;
                let mut best_cls = 0;

                // Check standard vehicle classes: 2: car, 3: motorcycle, 5: bus, 7: truck
                for &c in &[2usize, 3, 5, 7] {
                    let score = data[(4 + c) * num_anchors + i];
                    if score > max_cls_conf {
                        max_cls_conf = score;
                        best_cls = c;
                    }
                }

                if max_cls_conf >= conf_threshold {
                    let cx = data[0 * num_anchors + i] * x_scale;
                    let cy = data[1 * num_anchors + i] * y_scale;
                    let w = data[2 * num_anchors + i] * x_scale;
                    let h = data[3 * num_anchors + i] * y_scale;

                    let x1 = (cx - w / 2.0).max(0.0);
                    let y1 = (cy - h / 2.0).max(0.0);
                    let x2 = (cx + w / 2.0).min(width as f32);
                    let y2 = (cy + h / 2.0).min(height as f32);

                    let label = match best_cls {
                        2 => "car",
                        3 => "motorcycle",
                        5 => "bus",
                        7 => "truck",
                        _ => "vehicle",
                    };

                    boxes.push(([x1, y1, x2, y2], max_cls_conf, best_cls, label));
                }
            }

            boxes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut final_boxes = Vec::new();
            for b in boxes {
                let mut overlap = false;
                for fb in &final_boxes {
                    if calculate_iou(&b.0, &fb.0) > 0.45 {
                        overlap = true;
                        break;
                    }
                }
                if !overlap {
                    final_boxes.push(b);
                }
            }
            final_boxes
        });

        let py_list = PyList::empty_bound(py);
        for (bbox, conf, class_id, label) in detections {
            let dict = PyDict::new_bound(py);
            dict.set_item("class_id", class_id)?;
            dict.set_item("conf", conf)?;
            dict.set_item("bbox", vec![bbox[0], bbox[1], bbox[2], bbox[3]])?;
            dict.set_item("label", label)?;
            py_list.append(dict)?;
        }

        Ok(py_list.into())
    }

    pub fn detect_lanes_nn<'py>(
        &self,
        py: Python<'py>,
        frame: PyReadonlyArray3<'_, u8>,
    ) -> PyResult<PyObject> {
        let frame_arr = frame.as_array();
        if let Some(ref session) = self.lane_session {
            let lanes = lane_detect::detect_lanes_ufld(session, &frame_arr)
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

            let py_list = PyList::empty_bound(py);
            for lane in lanes {
                let point_list = PyList::empty_bound(py);
                for (x, y) in lane.points {
                    point_list.append((x, y))?;
                }
                py_list.append(point_list)?;
            }
            Ok(py_list.into())
        } else {
            let lines = lane_detect::detect_lanes(&frame_arr)
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

            let py_list = PyList::empty_bound(py);
            for line in lines {
                py_list.append(vec![(line.0, line.1), (line.2, line.3)])?;
            }
            Ok(py_list.into())
        }
    }

    pub fn detect_traffic_lights(
        &mut self,
        py: Python,
        frame_bytes: &[u8],
        width: u32,
        height: u32,
    ) -> PyResult<PyObject> {
        if let Some(ref mut detector) = self.light_detector {
            let results = detector.detect(frame_bytes, width, height);
            let py_list = PyList::empty_bound(py);

            for res in results {
                let dict = PyDict::new_bound(py);
                dict.set_item("bbox", vec![res.bbox[0], res.bbox[1], res.bbox[2], res.bbox[3]])?;
                dict.set_item("status", format!("{:?}", res.status))?;
                dict.set_item("voted_status", format!("{:?}", res.voted_status))?;
                dict.set_item("confidence", res.cls_confidence)?;
                py_list.append(dict)?;
            }
            Ok(py_list.into())
        } else {
            let py_list = PyList::empty_bound(py);
            Ok(py_list.into())
        }
    }

    pub fn detect_signs(
        &self,
        py: Python,
        frame_bytes: &[u8],
        width: u32,
        height: u32,
        conf_threshold: f32,
    ) -> PyResult<PyObject> {
        let detections = py.allow_threads(|| {
            let mut chw = vec![0.0f32; 3 * 320 * 320];
            let x_scale = width as f32 / 320.0;
            let y_scale = height as f32 / 320.0;

            for y in 0..320 {
                let src_y = ((y as f32 * y_scale) as u32).min(height - 1);
                for x in 0..320 {
                    let src_x = ((x as f32 * x_scale) as u32).min(width - 1);
                    let src_idx = ((src_y * width + src_x) * 3) as usize;
                    if src_idx + 2 < frame_bytes.len() {
                        chw[y * 320 + x] = frame_bytes[src_idx + 2] as f32 / 255.0;
                        chw[320 * 320 + y * 320 + x] = frame_bytes[src_idx + 1] as f32 / 255.0;
                        chw[2 * 320 * 320 + y * 320 + x] = frame_bytes[src_idx] as f32 / 255.0;
                    }
                }
            }

            let tensor = match Tensor::from_array(([1usize, 3, 320, 320], chw)) {
                Ok(t) => t,
                Err(_) => return Vec::new(),
            };

            let outputs = match self.sign_session.run(ort::inputs!["images" => tensor].unwrap()) {
                Ok(o) => o,
                Err(_) => return Vec::new(),
            };

            let (_, data) = match outputs[0].try_extract_tensor::<f32>() {
                Ok(d) => d,
                Err(_) => return Vec::new(),
            };

            let num_anchors = 2100;
            let num_classes = 4;
            let mut boxes = Vec::new();

            for i in 0..num_anchors {
                let mut max_cls_conf = 0.0f32;
                let mut best_cls = 0;

                for c in 0..num_classes {
                    let score = data[(4 + c) * num_anchors + i];
                    if score > max_cls_conf {
                        max_cls_conf = score;
                        best_cls = c;
                    }
                }

                if max_cls_conf >= conf_threshold {
                    let cx = data[0 * num_anchors + i] * x_scale;
                    let cy = data[1 * num_anchors + i] * y_scale;
                    let w = data[2 * num_anchors + i] * x_scale;
                    let h = data[3 * num_anchors + i] * y_scale;

                    let x1 = (cx - w / 2.0).max(0.0);
                    let y1 = (cy - h / 2.0).max(0.0);
                    let x2 = (cx + w / 2.0).min(width as f32);
                    let y2 = (cy + h / 2.0).min(height as f32);

                    boxes.push(([x1, y1, x2, y2], max_cls_conf, best_cls));
                }
            }

            boxes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut final_boxes = Vec::new();
            for b in boxes {
                let mut overlap = false;
                for fb in &final_boxes {
                    if calculate_iou(&b.0, &fb.0) > 0.45 {
                        overlap = true;
                        break;
                    }
                }
                if !overlap {
                    final_boxes.push(b);
                }
            }
            final_boxes
        });

        let py_list = PyList::empty_bound(py);
        for (bbox, conf, class_id) in detections {
            let dict = PyDict::new_bound(py);
            dict.set_item("class_id", class_id)?;
            dict.set_item("conf", conf)?;
            dict.set_item("bbox", vec![bbox[0], bbox[1], bbox[2], bbox[3]])?;
            py_list.append(dict)?;
        }

        Ok(py_list.into())
    }
}

fn calculate_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
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

#[pyclass]
pub struct Tracker {
    inner: ObjectTracker,
}

#[pymethods]
impl Tracker {
    #[new]
    pub fn new() -> Self {
        Tracker {
            inner: ObjectTracker::new(),
        }
    }

    pub fn process_frame(
        &mut self,
        detections: Vec<(f64, f64, f64, f64, String)>,
        dt: f64,
    ) -> Vec<(usize, f64, f64, f64, f64, f64, f64, f64, f64, f64, String)> {
        let results = self.inner.process_frame(detections, dt);
        results
            .into_iter()
            .map(|o| {
                (
                    o.id,
                    o.bbox.0,
                    o.bbox.1,
                    o.bbox.2,
                    o.bbox.3,
                    o.distance,
                    o.speed,
                    o.collisiontime,
                    o.velocity.0,
                    o.velocity.1,
                    o.class_label,
                )
            })
            .collect()
    }
}

#[pyclass]
pub struct LaneManager {
    inner: CoreLaneManager,
}

#[pymethods]
impl LaneManager {
    #[new]
    #[pyo3(signature = (smoothing=0.8, is_two_way=false))]
    pub fn new(smoothing: f64, is_two_way: bool) -> Self {
        LaneManager {
            inner: CoreLaneManager::new(smoothing, is_two_way),
        }
    }

    pub fn update_lines(
        &mut self,
        raw_lines: Vec<(f64, f64, f64, f64)>,
        img_width: f64,
    ) -> (Option<(f64, f64, f64, f64)>, Option<(f64, f64, f64, f64)>) {
        self.inner.update_lines(raw_lines, img_width)
    }

    pub fn check_departure(&self, img_width: f64, img_height: f64) -> String {
        match self.inner.check_departure(img_width, img_height) {
            lane_manager::DepartureWarning::None => "NONE".to_string(),
            lane_manager::DepartureWarning::DriftingLeft => "DRIFTING_LEFT".to_string(),
            lane_manager::DepartureWarning::DriftingRight => "DRIFTING_RIGHT".to_string(),
            lane_manager::DepartureWarning::DepartedLeft => "DEPARTED_LEFT".to_string(),
            lane_manager::DepartureWarning::DepartedRight => "DEPARTED_RIGHT".to_string(),
        }
    }
}

#[pyfunction]
pub fn detect_lanes(frame: PyReadonlyArray3<u8>) -> PyResult<Vec<(f64, f64, f64, f64)>> {
    lane_detect::detect_lanes(&frame.as_array()).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

#[pyfunction]
pub fn check_traffic_lights(frame: PyReadonlyArray3<u8>) -> PyResult<String> {
    let status = detect_traffic_light_hsv(&frame.as_array());
    Ok(format!("{:?}", status))
}

#[pymodule]
fn adas_pilot(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_lanes, m)?)?;
    m.add_function(wrap_pyfunction!(check_traffic_lights, m)?)?;

    m.add_class::<Tracker>()?;
    m.add_class::<LaneManager>()?;
    m.add_class::<AdasBrain>()?;

    Ok(())
}