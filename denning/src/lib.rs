#[cfg(feature = "charming")]
use charming::{
    Chart,
    component::{Axis, Title},
    element::{AxisType, Tooltip, Trigger},
    series::Line,
};
use core::f64;
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(feature = "plotters")]
use plotters::{coord::Shift, prelude::*};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::function::beta::beta_reg;

#[derive(Serialize, Deserialize)]
pub struct MissRatioCurve {
    miss_ratio: Box<[f64]>,
    turning_points: Box<[f64]>,
}

impl MissRatioCurve {
    /// Smith's set-associativity conversion, assuming a uniform set mapping
    /// (see the simulator's `--hash` mode): a reuse of distance `r` hits an
    /// `a`-way cache of `S = size/a` sets with probability
    /// `P[Binom(r, 1/S) <= a-1] = I_{1-1/S}(r-a+1, a)` — the regularized
    /// incomplete beta, continuous in both `r` and the cache size. For
    /// `S >> 1` this converges to the gamma/Poisson form
    /// `P[Pois(r/S) <= a-1] = Γ(a, r/S)/Γ(a)`; the beta form is kept because
    /// for small `S` the unbounded Poisson tail charges misses to reuses
    /// with fewer than `a` intervening blocks, which can never miss.
    pub fn compute_assoc(&self, associativity: usize) -> Self {
        let len = self.turning_points.len();
        let rd = &self.turning_points;
        // Step 1: Calculate RD distribution (q_j values) from miss ratios
        let mut rd_portions = vec![0.0; len];
        for i in 0..len {
            if i == 0 {
                rd_portions[i] = 1.0 - self.miss_ratio[i];
            } else {
                let tmp = 1.0 - self.miss_ratio[i];
                rd_portions[i] = tmp - (1.0 - self.miss_ratio[i - 1]);
            }
        }

        let max_tp_usize = rd.last().unwrap().ceil() as usize;
        let cache_sizes: Vec<_> = (associativity..(max_tp_usize + associativity))
            .step_by(1)
            .collect();

        // Calculate all miss ratios in parallel and show progress
        let pb = ProgressBar::new(cache_sizes.len() as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .tick_chars("◐◓◑◒ "),
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(std::thread::available_parallelism().unwrap().get())
            .build()
            .unwrap();

        let parallel_results: Vec<(f64, f64)> = pool.install(|| {
            cache_sizes
                .par_iter()
                .map_with(pb.clone(), |pb, &cache_size| {
                    // A single-set a-way LRU cache IS an a-block fully
                    // associative cache: degenerate to the input curve
                    // instead of Poisson-smearing it.
                    if cache_size == associativity {
                        pb.inc(1);
                        return (self.miss_ratio_at(cache_size as f64), cache_size as f64);
                    }
                    let mut miss_ratio = 0.0;
                    let number_of_sets = cache_size as f64 / associativity as f64;

                    let a = associativity as f64;
                    for (r, w) in rd.iter().copied().zip(rd_portions.iter().copied()) {
                        if r <= a - 1.0 {
                            // at most r blocks can intervene: guaranteed hit
                            continue;
                        }
                        // P[Binom(r, 1/S) <= a-1] = I_{1-1/S}(r-a+1, a)
                        let hit_prob = beta_reg(r - a + 1.0, a, 1.0 - 1.0 / number_of_sets);
                        miss_ratio += w * (1.0 - hit_prob);
                    }

                    pb.inc(1);
                    (miss_ratio, cache_size as f64)
                })
                .collect()
        });

        pb.finish_and_clear();

        // Create the final vectors with initial values
        let mut new_miss_ratio = parallel_results
            .iter()
            .map(|(ratio, _)| *ratio)
            .collect::<Vec<_>>();
        let mut new_turning_points = parallel_results
            .iter()
            .map(|(_, size)| *size)
            .collect::<Vec<_>>();

        // find where miss ratio becomes negative, and remove those results, and repective turning points
        if let Some(pos) = new_miss_ratio.iter().position(|&x| x < 0.0) {
            new_miss_ratio.truncate(pos);
            new_turning_points.truncate(pos);
        }

        Self {
            miss_ratio: new_miss_ratio.into_boxed_slice(),
            turning_points: new_turning_points.into_boxed_slice(),
        }
    }

    pub fn new(ri_dist: &[(isize, f64)]) -> Self {
        let mut miss_ratio = ri_dist.iter().map(|(_, prob)| *prob).collect::<Vec<_>>();
        let mut rolling_sum = 1.0 - miss_ratio.iter().sum::<f64>();
        for i in miss_ratio.iter_mut().rev() {
            let current = *i;
            *i = rolling_sum;
            rolling_sum += current;
        }
        let mut turning_points = vec![0.0; miss_ratio.len()];
        let mut prev = 0.0;
        for (i, iter) in turning_points.iter_mut().enumerate().skip(1) {
            *iter = prev + miss_ratio[i - 1] * (ri_dist[i].0 - ri_dist[i - 1].0) as f64;
            prev = *iter;
        }
        let miss_ratio = miss_ratio.into_boxed_slice();
        let turning_points = turning_points.into_boxed_slice();
        Self {
            miss_ratio,
            turning_points,
        }
    }

    /// Miss ratio at each turning point, in the same order as
    /// [`Self::turning_points`].
    pub fn miss_ratio(&self) -> &[f64] {
        &self.miss_ratio
    }

    /// Cache sizes at which the miss ratio steps down.
    pub fn turning_points(&self) -> &[f64] {
        &self.turning_points
    }

    /// Miss ratio for a fully-associative LRU cache of `cache_size` blocks.
    ///
    /// The curve is a step function, so this is the miss ratio attached to the
    /// last turning point at or below `cache_size`. Sizes below the first
    /// turning point miss everything; an empty curve reports a miss ratio of 1
    /// rather than dividing into nothing.
    pub fn miss_ratio_at(&self, cache_size: f64) -> f64 {
        if self.turning_points.is_empty() {
            return 1.0;
        }
        // Turning points are non-decreasing, so a binary search finds the step
        // containing `cache_size` without scanning a long curve per query.
        let index = self
            .turning_points
            .partition_point(|point| *point <= cache_size);
        match index.checked_sub(1) {
            Some(index) => self.miss_ratio[index],
            None => 1.0,
        }
    }

    #[cfg(feature = "plotters")]
    pub fn plot_miss_ratio_curve<DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, Shift>,
    ) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
        let turning_points = &self.turning_points;
        let miss_ratio = &self.miss_ratio;
        area.fill(&WHITE)?;
        let max_x = *turning_points.last().unwrap_or(&1.0);

        let mut chart = ChartBuilder::on(area)
            .caption("Miss Ratio Curve", ("sans-serif", 40))
            .margin(50)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d((0.0..max_x).log_scale(), 0.0..1.0)?;

        chart
            .configure_mesh()
            .x_desc("Cache Size")
            .y_desc("Miss Ratio")
            .draw()?;

        let mut points = Vec::new();

        for i in 0..miss_ratio.len() {
            points.push((turning_points[i], miss_ratio[i]));
            if i + 1 < turning_points.len() {
                points.push((turning_points[i + 1], miss_ratio[i]));
            }
        }

        chart
            .draw_series(LineSeries::new(points, &RED))?
            .label("Miss Ratio")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

        chart.configure_series_labels().border_style(BLACK).draw()?;

        Ok(())
    }

    #[cfg(feature = "charming")]
    pub fn plot_interactive_miss_ratio_curve(&self) -> Chart {
        use charming::element::Step;

        let turning_points = &self.turning_points;
        let miss_ratio = &self.miss_ratio;

        //--------------------------------------------------------------------
        // Build coordinates so ECharts can display a true *step* line.
        // Each turning-point is duplicated: first at the old height, then at
        // the new one – this creates the vertical drop.
        //--------------------------------------------------------------------
        let mut coords: Vec<Vec<f64>> = Vec::with_capacity(turning_points.len());
        for i in 0..turning_points.len() {
            let x = turning_points[i];
            let y = miss_ratio[i];
            coords.push(vec![x, y]); // horizontal
        }

        //--------------------------------------------------------------------
        // Compose the chart.
        //--------------------------------------------------------------------
        Chart::new()
            .title(Title::new().text("Miss-ratio curve"))
            .x_axis(
                Axis::new()
                    .name("Cache size")
                    .type_(AxisType::Value) // ← logarithmic axis :contentReference[oaicite:0]{index=0}
                    .min(0.0)
                    .max(Some(*turning_points.last().unwrap()))
                    .scale(true)
                    .log_base(turning_points.last().unwrap().ceil()), // use base-10 ticks; change if you prefer
            )
            .y_axis(
                Axis::new()
                    .name("Miss ratio")
                    .min(0.0)
                    .type_(AxisType::Value)
                    .max(1.0),
            )
            .tooltip(Tooltip::new().trigger(Trigger::Axis)) // hover shows (x, y)
            .series(
                Line::new()
                    .name("Miss ratio")
                    .data(coords) // (x, y) coordinate pairs
                    .step(Step::End) // horizontal → drop at the *end* of an interval
                    .show_symbol(false),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "0 1 2 2 1 0" has reuse intervals 1, 3 and 5, one access each out of
    /// six, and three compulsory misses.
    fn example() -> MissRatioCurve {
        MissRatioCurve::new(&[(0, 0.0), (1, 1.0 / 6.0), (3, 1.0 / 6.0), (5, 1.0 / 6.0)])
    }

    #[test]
    fn miss_ratio_is_non_increasing_in_cache_size() {
        let curve = example();
        let mut previous = f64::INFINITY;
        for size in 0..40 {
            let observed = curve.miss_ratio_at(f64::from(size));
            assert!(
                observed <= previous + 1e-12,
                "miss ratio rose from {previous} to {observed} at cache size {size}"
            );
            previous = observed;
        }
    }

    #[test]
    fn a_cache_below_the_first_turning_point_misses_everything() {
        assert_eq!(example().miss_ratio_at(-1.0), 1.0);
    }

    #[test]
    fn an_empty_curve_reports_total_misses() {
        assert_eq!(MissRatioCurve::new(&[]).miss_ratio_at(1024.0), 1.0);
    }

    #[test]
    fn a_large_cache_leaves_only_the_compulsory_misses() {
        let curve = example();
        let largest = curve.turning_points().last().copied().unwrap_or_default();
        // Three of six accesses are compulsory misses.
        assert!(
            (curve.miss_ratio_at(largest + 1000.0) - 0.5).abs() < 1e-9,
            "got {}",
            curve.miss_ratio_at(largest + 1000.0)
        );
    }

    #[test]
    fn lookup_agrees_with_a_linear_scan() {
        let curve = example();
        for size in 0..40 {
            let size = f64::from(size);
            let expected = curve
                .turning_points()
                .iter()
                .zip(curve.miss_ratio())
                .filter(|(point, _)| **point <= size)
                .next_back()
                .map_or(1.0, |(_, ratio)| *ratio);
            assert_eq!(curve.miss_ratio_at(size), expected, "at cache size {size}");
        }
    }
}
