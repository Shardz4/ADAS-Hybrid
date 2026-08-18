#[derive(Clone, Debug)]
pub struct Mat4(pub [f64;16]);

impl Mat4 {
    #[inline]
    pub fn identity() -> Self {
        Mat4([
            1.0,0.0,0.0,0.0,
            0.0,1.0,0.0,0.0,
            0.0,0.0,1.0,0.0,
            0.0,0.0,0.0,1.0,
        ])
    }

    #[inline]
    pub fn zero() -> Self {
        Mat4([0.0; 16])
    }

    #[inline]
    pub fn mul(&self, other: &Mat4) -> Mat4 {
        let mut out = [0.0; 16];
        for r in 0..4{
            for c in 0..4{
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.0[r*4 + k] * other.0[k*4 +c];
                }
                out[r*4 + c] = sum;
            }
        }
        Mat4(out)
    }

    #[inline]
    pub fn add(&self, other:&Mat4) -> Mat4 {
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
            m[0]*v[0] + m[1]*v[1] + m[2]*v[2] + m[3]*v[3],
            m[4]*v[0] + m[5]*v[1] + m[6]*v[2] + m[7]*v[3],
            m[8]*v[0] + m[9]*v[1] + m[10]*v[2] + m[11]*v[3],
            m[12]*v[0] + m[13]*v[1] + m[14]*v[2] + m[15]*v[3],
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
    pub h:Mat2x4,
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
        let h = Mat2x4({
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 100.0].
        });

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

    pub fn predict(&mut self, dt:f64) -> [f64; 4] {
        self.f.0[2] = dt;
        self.f.0[7] = dt;
        self.x = self.f.mul_vec(&self.x);
        let ft = self.f.transpose();
        let fp = self.f.mul(&self.p);
        self.p = fp,mul(&ft).add(&self.q);

        self.x
    }
    

    
}