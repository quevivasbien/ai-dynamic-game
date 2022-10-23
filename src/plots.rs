use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;

use numpy::ndarray::{s};

use crate::utils::{range_from_iter};
use crate::strategies::*;

const X_RANGE_BUFFER: f64 = 0.;
const Y_RANGE_BUFFER: f64 = 0.1;

pub struct PlotOptions {
    x_range: Option<(f64, f64)>,
    y_range: Option<(f64, f64)>,
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
}

impl Default for PlotOptions {
    fn default() -> Self {
        Self {
            x_range: None,
            y_range: None,
            title: None,
            x_label: None,
            y_label: None,
        }
    }
}

type Ctx<'a> = ChartContext<'a, BitMapBackend<'a>, Cartesian2d<RangedCoordf64, RangedCoordf64>>;

pub fn line_plot<'a, 'b, I, J>(mut x: I, mut y: J, filename: &'b str, options: &PlotOptions) -> Result<Ctx<'b>, Box<dyn std::error::Error>>
where
    I: Iterator<Item = &'a f64>,
    J: Iterator<Item = &'a f64>,
{
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let title = options.title.clone().unwrap_or("".to_string());
    let xrange = match options.x_range {
        Some((x_min, x_max)) => x_min..x_max,
        None => range_from_iter(&mut x, X_RANGE_BUFFER),
    };
    let yrange = match options.y_range {
        Some((y_min, y_max)) => y_min..y_max,
        None => range_from_iter(&mut y, Y_RANGE_BUFFER),
    };
    
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(title, ("sans-serif", 50).into_font())
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(xrange, yrange)?;
    
    let x_label = options.x_label.clone().unwrap_or("x".to_string());
    let y_label = options.y_label.clone().unwrap_or("y".to_string());

    chart
        .configure_mesh()
        // .disable_x_mesh()
        // .disable_y_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .axis_desc_style(("sans-serif", 15).into_font())
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            x.zip(y).map(|(x_, y_)| (*x_, *y_)),
            &RED,
        ))?
        .label("Line")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    Ok(chart)
}

pub fn add_series<'a, 'b, I, S>(x: I, y: I, mut chart: Ctx, style: &S) -> Result<Ctx<'b>, Box<dyn std::error::Error>>
where
    I: Iterator<Item = &'a f64>,
    S: Into<ShapeStyle>,
{
    chart
        .draw_series(LineSeries::new(
            x.zip(y).map(|(x_, y_)| (*x_, *y_)),
            *style,
        ))?
        .label("Line")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], *style));

    Ok(chart)
}

pub trait Plottable {
    fn plot<'a>(&self, filename: &str, options: &PlotOptions) -> Result<Ctx<'a>, Box<dyn std::error::Error>>;
}

impl Plottable for Strategies {
    fn plot<'a>(&self, filename: &str, options: &PlotOptions) -> Result<Ctx<'a>, Box<dyn std::error::Error>> {
        let chart = line_plot(
            (0..self.t()).map(|t| &(t as f64)),
            self.data().slice(s![.., 0, 0]).iter(),
            "test.png", &PlotOptions::default()
        )?;
        Ok(chart)
    }
}