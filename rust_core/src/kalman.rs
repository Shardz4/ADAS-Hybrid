#[derive(Clone, Debug)]
pub struct Mat4(pub [f64; 16]);

impl Mat4 {
    #[inline]
    pub fn identity() -> Self {
        Mat4([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    #[inline]
    pub fn zero() -> Self {
        Mat4([0.0; 16])
    }

    #[inline]
    pub fn mul(&self, other: &Mat4) -> Mat4 {
        let mut out = [0.0; 16];
        for r in 0..4 {
            for c in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.0[r * 4 + k] * other.0[k * 4 + c];
                }
                out[r * 4 + c] = sum;
            }
        }
        Mat4(out)
    }

    #[inline]
    pub fn add(&self, other: &Mat4) -> Mat4 {
        let mut out = [0.0; 16];
        for i in 0..16 {
            out[i] = self.0[i] + other.0[i];
        }
        Mat4(out)
    }

    #[inline]
    pub fn transpose(&self) -> Mat4 {
        let m = &self.0;
        Mat4([
            m[0], m[4], m[8], m[12],
            m[1], m[5], m[9], m[13],
            m[2], m[6], m[10], m[14],
            m[3], m[7], m[11], m[15],
        ])
    }

    #[inline]
    pub fn mul_vec(&self, v: &[f64; 4]) -> [f64; 4] {
        let m = &self.0;
        [
            m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3] * v[3],
            m[4] * v[0] + m[5] * v[1] + m[6] * v[2] + m[7] * v[3],
            m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11] * v[3],
            m[12] * v[0] + m[13] * v[1] + m[14] * v[2] + m[15] * v[3],
        ]
    }
}

#[derive(Clone, Debug)]
pub struct Mat2x4(pub [[f64; 4]; 2]);

#[derive(Clone, Debug)]
pub struct Mat4x2(pub [[f64; 2]; 4]);

#[derive(Clone, Debug)]
pub struct KalmanFilter2D {
    pub x: [f64; 4],
    pub p: Mat4,
    pub f: Mat4,
    pub h: Mat2x4,
    pub q: Mat4,
    pub r: [[f64; 2]; 2],
}

impl KalmanFilter2D {
    pub fn new(initial_cx: f64, initial_cy: f64) -> Self {
        let p = Mat4([
            50.0, 0.0, 0.0, 0.0,
            0.0, 50.0, 0.0, 0.0,
            0.0, 0.0, 100.0, 0.0,
            0.0, 0.0, 0.0, 100.0,
        ]);

        let f = Mat4::identity();

        // Observation matrix: maps state [x, y, vx, vy] to measurement [x, y]
        let h = Mat2x4([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]);

        let q = Mat4([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.0, 0.5,
        ]);

        let r = [
            [5.0, 0.0],
            [0.0, 5.0],
        ];

        KalmanFilter2D {
            x: [initial_cx, initial_cy, 0.0, 0.0],
            p,
            f,
            h,
            q,
            r,
        }
    }

    pub fn predict(&mut self, dt: f64) -> [f64; 4] {
        // Update transition matrix with current dt
        self.f.0[2] = dt;  // F[0][2] = dt
        self.f.0[7] = dt;  // F[1][3] = dt

        // x = F * x
        self.x = self.f.mul_vec(&self.x);

        // P = F * P * F^T + Q
        let ft = self.f.transpose();
        let fp = self.f.mul(&self.p);
        self.p = fp.mul(&ft).add(&self.q);

        self.x
    }

    pub fn update(&mut self, measurement: [f64; 2]) {
        // y = z - H * x  (innovation)
        let hx = [
            self.h.0[0][0] * self.x[0] + self.h.0[0][1] * self.x[1]
                + self.h.0[0][2] * self.x[2] + self.h.0[0][3] * self.x[3],
            self.h.0[1][0] * self.x[0] + self.h.0[1][1] * self.x[1]
                + self.h.0[1][2] * self.x[2] + self.h.0[1][3] * self.x[3],
        ];

        let y = [measurement[0] - hx[0], measurement[1] - hx[1]];

        // S = H * P * H^T + R  (innovation covariance)
        // First compute HP = H * P  (2x4)
        let mut hp = [[0.0; 4]; 2];
        for r in 0..2 {
            for c in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.h.0[r][k] * self.p.0[k * 4 + c];
                }
                hp[r][c] = sum;
            }
        }

        // Then S = HP * H^T + R  (2x2)
        let mut s = [[0.0; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += hp[r][k] * self.h.0[c][k];
                }
                s[r][c] = sum + self.r[r][c];
            }
        }

        // Invert S (2x2)
        let det = s[0][0] * s[1][1] - s[0][1] * s[1][0];
        if det.abs() < 1e-9 {
            return;
        }
        let inv_det = 1.0 / det;
        let s_inv = [
            [s[1][1] * inv_det, -s[0][1] * inv_det],
            [-s[1][0] * inv_det, s[0][0] * inv_det],
        ];

        // K = P * H^T * S^-1  (4x2)
        // First compute PH^T  (4x2)
        let mut pht = [[0.0; 2]; 4];
        for r in 0..4 {
            for c in 0..2 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.p.0[r * 4 + k] * self.h.0[c][k];
                }
                pht[r][c] = sum;
            }
        }

        // Then K = PH^T * S^-1  (4x2)
        let mut k_gain = [[0.0; 2]; 4];
        for r in 0..4 {
            for c in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += pht[r][k] * s_inv[k][c];
                }
                k_gain[r][c] = sum;
            }
        }

        // x = x + K * y
        self.x[0] += k_gain[0][0] * y[0] + k_gain[0][1] * y[1];
        self.x[1] += k_gain[1][0] * y[0] + k_gain[1][1] * y[1];
        self.x[2] += k_gain[2][0] * y[0] + k_gain[2][1] * y[1];
        self.x[3] += k_gain[3][0] * y[0] + k_gain[3][1] * y[1];

        // P = (I - K * H) * P
        let mut kh = Mat4::zero();
        for r in 0..4 {
            for c in 0..4 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += k_gain[r][k] * self.h.0[k][c];
                }
                kh.0[r * 4 + c] = sum;
            }
        }

        let mut i_minus_kh = Mat4::identity();
        for i in 0..16 {
            i_minus_kh.0[i] -= kh.0[i];
        }
        self.p = i_minus_kh.mul(&self.p);
    }

    #[inline]
    pub fn state(&self) -> (f64, f64, f64, f64) {
        (self.x[0], self.x[1], self.x[2], self.x[3])
    }

    #[inline]
    pub fn peek_predict(&self, dt: f64) -> (f64, f64) {
        (self.x[0] + self.x[2] * dt, self.x[1] + self.x[3] * dt)
    }
}