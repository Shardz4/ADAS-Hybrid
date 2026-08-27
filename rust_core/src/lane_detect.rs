use ndarray::ArrayView3;
use ort::session::Session;
use ort::value::Tensor;

pub type Line = (f64, f64, f64, f64);

#[derive(Clone, Debug)]
pub struct LanePolyline {
    pub points: Vec<(f64, f64)>,
    pub confidence: f32,
    pub lane_index: i32,
}

pub fn preprocess_ufld(frame: &ArrayView3<u8>) -> Result<Vec<f32>, String> {
    let (h, w, c) = (frame.dim().0, frame.dim().1, frame.dim().2);
    if c != 3 {
        return Err("Input frame must have 3 color channels".to_string());
    }

    let target_h = 288;
    let target_w = 800;
    let mut chw = vec![0.0f32; 3 * target_h * target_w];

    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];

    let y_scale = h as f32 / target_h as f32;
    let x_scale = w as f32 / target_w as f32;

    for y in 0..target_h {
        let src_y = ((y as f32 * y_scale) as usize).min(h - 1);
        for x in 0..target_w {
            let src_x = ((x as f32 * x_scale) as usize).min(w - 1);

            let b = frame[[src_y, src_x, 0]] as f32 / 255.0;
            let g = frame[[src_y, src_x, 1]] as f32 / 255.0;
            let r = frame[[src_y, src_x, 2]] as f32 / 255.0;

            let idx = y * target_w + x;
            chw[idx] = (r - mean[0]) / std[0];
            chw[target_h * target_w + idx] = (g - mean[1]) / std[1];
            chw[2 * target_h * target_w + idx] = (b - mean[2]) / std[2];
        }
    }
    Ok(chw)
}

pub fn decode_ufld_output(
    output: &[f32],
    num_lanes: usize,
    num_row_anchors: usize,
    num_grid_cells: usize,
    original_h: u32,
    original_w: u32,
) -> Vec<LanePolyline> {
    let classes_per_row = num_grid_cells + 1;
    let mut polylines = Vec::with_capacity(num_lanes);

    let row_anchor_start = 160.0;
    let row_anchor_end = 284.0;
    let row_step = (row_anchor_end - row_anchor_start) / (num_row_anchors - 1) as f64;

    for lane_idx in 0..num_lanes {
        let lane_offset = lane_idx * num_row_anchors * classes_per_row;
        let mut points = Vec::new();

        for row in 0..num_row_anchors {
            let row_offset = lane_offset + row * classes_per_row;
            let cell_slice = &output[row_offset..row_offset + classes_per_row];

            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for (idx, &val) in cell_slice.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = idx;
                }
            }

            if max_idx < num_grid_cells {
                let norm_x = max_idx as f64 / num_grid_cells as f64;
                let actual_x = norm_x * original_w as f64;

                let norm_y = (row_anchor_start + row as f64 * row_step) / 288.0;
                let actual_y = norm_y * original_h as f64;

                points.push((actual_x, actual_y));
            }
        }

        points.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        polylines.push(LanePolyline {
            points,
            confidence: 1.0,
            lane_index: lane_idx as i32 - 1,
        });
    }
    polylines
}

pub fn detect_lanes_ufld(
    session: &mut Session,
    frame: &ArrayView3<u8>,
) -> Result<Vec<LanePolyline>, String> {
    let input = preprocess_ufld(frame)?;
    let tensor = Tensor::from_array(([1usize, 3, 288, 800], input))
        .map_err(|e| format!("Failed to create UFLD input tensor: {}", e))?;

    let outputs = session
        .run(ort::inputs!["input" => tensor])
        .map_err(|e| format!("UFLD inference failed: {}", e))?;

    let (_, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("Failed to extract UFLD output tensor: {}", e))?;

    let lanes = decode_ufld_output(
        data,
        4,
        56,
        100,
        frame.dim().0 as u32,
        frame.dim().1 as u32,
    );
    Ok(lanes.into_iter().filter(|l| l.points.len() >= 5).collect())
}

pub fn polylines_to_lines(lanes: &[LanePolyline]) -> Vec<Line> {
    lanes
        .iter()
        .filter_map(|l| {
            if l.points.len() >= 2 {
                let first = l.points.first().unwrap();
                let last = l.points.last().unwrap();
                Some((first.0, first.1, last.0, last.1))
            } else {
                None
            }
        })
        .collect()
}

pub fn detect_lanes(frame: &ArrayView3<u8>) -> Result<Vec<Line>, String> {
    let gray = bgr_to_gray(frame);
    let blurred = apply_gaussian_blur(&gray);
    let roi = apply_roi(&blurred, gray.width(), gray.height());
    let edges = detect_edges(&roi);
    let lines = hough_transform(&edges);
    Ok(lines)
}

fn bgr_to_gray(frame: &ArrayView3<u8>) -> image::GrayImage {
    let (h, w) = (frame.dim().0 as u32, frame.dim().1 as u32);
    let mut gray = image::GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let b = frame[[y as usize, x as usize, 0]] as f32;
            let g = frame[[y as usize, x as usize, 1]] as f32;
            let r = frame[[y as usize, x as usize, 2]] as f32;
            let lum = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
            gray.put_pixel(x, y, image::Luma([lum]));
        }
    }
    gray
}

fn apply_gaussian_blur(img: &image::GrayImage) -> image::GrayImage {
    imageproc::filter::gaussian_blur_f32(img, 1.5)
}

fn apply_roi(img: &image::GrayImage, width: u32, height: u32) -> image::GrayImage {
    let mut mask = image::GrayImage::new(width, height);
    for y in (height / 2)..height {
        for x in 0..width {
            mask.put_pixel(x, y, *img.get_pixel(x, y));
        }
    }
    mask
}

fn detect_edges(img: &image::GrayImage) -> image::GrayImage {
    imageproc::edges::canny(img, 50.0, 150.0)
}

fn hough_transform(edges: &image::GrayImage) -> Vec<Line> {
    let mut lines = Vec::new();
    let (w, h) = (edges.width() as f64, edges.height() as f64);

    // Placeholder: return two representative ego-lane lines
    lines.push((w * 0.2, h, w * 0.45, h * 0.6));
    lines.push((w * 0.8, h, w * 0.55, h * 0.6));
    lines
}
