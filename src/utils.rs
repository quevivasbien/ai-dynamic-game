use numpy::ndarray::{ArrayView, Dimension};

pub fn isapprox(a: f64, b: f64, rtol: f64, atol: f64) -> bool
{
    let maxval = f64::max(a.abs(), b.abs());
    (a - b).abs() <= f64::max(atol, rtol * maxval)
}

pub fn isapprox_arr<D>(a: ArrayView<f64, D>, b: ArrayView<f64, D>, rtol: f64, atol: f64) -> bool
where D: Dimension
{
    a.iter().zip(b.iter()).all(|(a_, b_)| isapprox(*a_, *b_, rtol, atol))
}
