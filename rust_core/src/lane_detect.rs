use ndarray::ArrayView3;
use ort::session::Session;
use ort::value::Tensor;

pub type Line = (f64, f64, f64, f64);

#[derive(Clone, Debug)]
pub struct LanePolyLIne {
    pub points: Vec<(f64, f64)>,
    pub confidence: f32,
    pub lane_index: i32,
}

pub fn preprocess_ufld(frame: &ArrayView3<u8>) -> Result<Vec<f32>, String> {
    let (h, w, c) = (frame.dim().0, frame.dim().1, frame.dim().2);
    if c!= 3 {
        return Err("Input frame must have 3 color channels".to_string());
    }

    let target_h = 288;
    let target_w = 800;
    let mut chw = vec![0.0f32; 3*target_h * target_w];

    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];

    let y_scale = h as f32 / target_h as f32;
    let x_scale = w as f32 / target_w as f32;

    for y in 0..target_h {
        let src_y = ((y as f32 * y_scale) as usize).min(h-1);
        for x in 0..target_w {
            let src_x = ((x as f32 * x_scale) as usize).min(w-1);

            let b = frame[[src_y, src_x, 0]] as f32 / 255.0;
            let g = frame[[src_y, src_x, 1]] as f32 / 255.0;
            let r = frame[[src_y, src_x, 2]] as f32 / 255.0;

            let idx = y * target_w + x;
            chw[idx] = (r-mean[0]) / std[0];
            chw[target_h * target_w + idx] = (g-mean[1]) / std[1];
            chw[2 * target_h * target_w + idx] = (b-mean[2]) / std[2];

        }
    }
    Ok(chw)
}