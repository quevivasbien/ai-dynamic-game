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

pub fn maximum<'a, I>(xs: &mut I) -> f64
where
    I: Iterator<Item = &'a f64>,
{
    xs.fold(f64::NEG_INFINITY, |acc, x| f64::max(acc, *x))
}

pub fn minimum<'a, I>(xs: &mut I) -> f64
where
    I: Iterator<Item = &'a f64>,
{
    xs.fold(f64::INFINITY, |acc, x| f64::min(acc, *x))
}

pub fn range_from_iter<'a, I>(xs: &mut I, buffer: f64) -> std::ops::Range<f64>
where
    I: Iterator<Item = &'a f64>,
{
    let min = minimum(xs);
    let max = maximum(xs);
    let diff = max - min;
    min - diff * buffer..max + diff * buffer
}
