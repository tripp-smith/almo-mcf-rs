pub mod barrier;
mod gradient;
mod vec_ops;

pub use gradient::{duality_gap_proxy, gradient_proxy};
pub use vec_ops::{dot, l1_norm, l2_norm, scaled_add};

pub const EPSILON: f64 = 1e-9;
pub const GRADIENT_EPSILON: f64 = 1e-12;
