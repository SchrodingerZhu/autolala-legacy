use anyhow::anyhow;
use barvinok::ContextRef as BContext;
use barvinok::constraint::Constraint;
use barvinok::local_space::LocalSpace;
use melior::Context as MContext;
use melior::ir::{BlockLike, Module, OperationRef, RegionLike, operation::OperationLike};
use palc::{Parser, Subcommand};
use plotters::prelude::IntoDrawingArea;
use raffine::Context as RContext;
use raffine::{DominanceInfo, tree::Tree};
use std::num::NonZero;
use std::{collections::HashMap, io::Read, path::PathBuf};
use tracing::{debug, error, info};
mod isl;
mod parallel;
mod polygeist;
mod salt;
mod utils;

use utils::create_table;
struct AnalysisContext<'a> {
    rcontext: RContext,
    bcontext: BContext<'a>,
}

impl<'a> AnalysisContext<'a>
where
    Self: 'a,
{
    fn start<F, R>(f: F) -> R
    where
        F: for<'x> FnOnce(AnalysisContext<'x>) -> R,
    {
        let rcontext = RContext::new();
        barvinok::Context::new().scope(move |bcontext| {
            let context = AnalysisContext { rcontext, bcontext };
            f(context)
        })
    }
    fn start_with_args<S: AsRef<str>, F, R>(args: &[S], f: F) -> anyhow::Result<R>
    where
        F: for<'x> FnOnce(AnalysisContext<'x>) -> anyhow::Result<R>,
    {
        let rcontext = RContext::new();
        unsafe { barvinok::Context::from_args(args.iter().map(|x| x.as_ref()))? }.scope(
            move |bcontext| {
                let context = AnalysisContext { rcontext, bcontext };
                f(context)
            },
        )
    }
    fn rcontext(&self) -> &RContext {
        &self.rcontext
    }
    fn bcontext(&self) -> BContext<'a> {
        self.bcontext
    }
    fn mcontext(&self) -> &MContext {
        self.rcontext.mlir_context()
    }
}

#[derive(Debug, Subcommand)]
enum Method {
    /// Use the Barvinok library to compute the polyhedral model
    Barvinok {
        #[arg(short = 'B', long)]
        /// barvinok options
        barvinok_arg: Vec<String>,
        #[arg(short = 'b', long, default_value = "1")]
        block_size: usize,
        #[arg(short = 'l', long)]
        symbol_lowerbound: Vec<i32>,
        #[arg(long)]
        infinite_repeat: bool,
        #[arg(short = 's', long, default_value = "1")]
        num_sets: NonZero<usize>,
        /// Abort isl computations after this many operations (0 = unlimited).
        /// The default keeps the historical unlimited behavior.
        #[arg(long, default_value = "0")]
        max_operations: usize,
        #[arg(long)]
        /// path to save the raw distribution bincode
        save_distro: Option<PathBuf>,
        /// Nesting level of the loop to model as `omp parallel for`
        /// (outermost is 0). Without this the analysis is sequential.
        #[arg(short = 'P', long)]
        parallel_loop_depth: Option<usize>,
        /// Thread count for the parallel model.
        #[arg(short = 'T', long, default_value = "1")]
        threads: NonZero<u32>,
        /// `schedule(static, chunk)` chunk size. Defaults to
        /// `ceil(trip / threads)`, OpenMP's own `schedule(static)`, when the
        /// parallel loop's bounds are compile-time constants.
        #[arg(short = 'C', long)]
        chunk: Option<NonZero<i64>>,
        /// Equal-probability buckets used to discretize the racetrack law.
        #[arg(long, default_value = "512")]
        racetrack_bins: NonZero<usize>,
        /// Tail mass below which a negative-binomial expansion stops.
        #[arg(long, default_value = "1e-12")]
        nbd_epsilon: f64,
        /// Reuse intervals at or below this use the negative binomial; longer
        /// ones use the concentrated law. Defaults to Theorem 3.1's bound.
        #[arg(long)]
        short_ri_bound: Option<f64>,
    },
    /// Use the PerfectTiling algorithm to compute the polyhedral model
    Salt {
        #[arg(short = 'b', long)]
        /// block size, if not specified, it will be represented symbolically
        block_size: Option<usize>,
    },
}

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug, Parser)]
struct Options {
    /// The input file to process
    /// If not specified, the input will be read from stdin
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// The output file to write
    /// If not specified, the output will be written to stdout
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// name of the target function to extract
    /// if not specified, the program will try to find first function
    #[arg(short = 'f', long)]
    target_function: Option<String>,

    /// target affine loop attribute
    /// if not specified, the program will try to find first affine loop in the function
    #[arg(short = 'l', long)]
    target_affine_loop: Option<String>,

    /// Miss ratio curve output path
    #[arg(short = 'm', long)]
    miss_ratio_curve: Option<PathBuf>,

    /// Miss ratio curve width
    /// if not specified, the default value is 800
    #[arg(short = 'W', long, default_value = "800")]
    miss_ratio_curve_width: u32,

    /// Miss ratio curve height
    /// if not specified, the default value is 600
    #[arg(short = 'H', long, default_value = "600")]
    miss_ratio_curve_height: u32,

    /// Use bincode encoded output
    /// Requires output file to be specified
    #[arg(long)]
    json: bool,

    #[arg(short = 'A', long, default_value = "1")]
    associativity: NonZero<usize>,

    /// method to use for polyhedral model computation
    #[command(subcommand)]
    method: Method,

    #[arg(long)]
    start_from_loop: bool,

    /// Treat the input as C source: lower it with Polygeist's `cgeist`
    /// (resolved from PATH) before analysis. Implied when --input ends in
    /// .c/.cc/.cpp/.cxx. Requires --input (cgeist reads a file).
    #[arg(long)]
    c_input: bool,
}

impl Options {
    fn wants_c_input(&self) -> bool {
        self.c_input
            || self
                .input
                .as_ref()
                .and_then(|p| p.extension())
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e, "c" | "cc" | "cpp" | "cxx"))
    }
}

/// Reads the analysis input as MLIR text; for C input, runs the Polygeist
/// bridge first. Returns the text and whether it came from Polygeist (in
/// which case the parsed module still carries foreign attributes to strip).
fn obtain_mlir_source(options: &Options, reader: &mut dyn Read) -> anyhow::Result<(String, bool)> {
    if options.wants_c_input() {
        let path = options
            .input
            .as_ref()
            .ok_or_else(|| anyhow!("--c-input requires --input: cgeist reads a file, not stdin"))?;
        let text = polygeist::run_polygeist(path, options.target_function.as_deref())
            .map_err(|e| anyhow!("polygeist bridge failed: {e}"))?;
        debug!("Polygeist produced {} bytes of MLIR", text.len());
        Ok((text, true))
    } else {
        let mut source = String::new();
        let bytes = reader.read_to_string(&mut source)?;
        debug!("Read {} bytes", bytes);
        Ok((source, false))
    }
}

fn extract_target<'bctx, 'ctx, 'dom>(
    module: &'ctx Module<'ctx>,
    options: &Options,
    context: &'ctx AnalysisContext<'bctx>,
    dom: &'ctx DominanceInfo<'ctx>,
) -> anyhow::Result<&'ctx Tree<'ctx>>
where
    'bctx: 'ctx,
{
    let body = module.body();
    fn locate_function<'a, 'b, F>(
        cursor: Option<OperationRef<'a, 'b>>,
        options: &'_ Options,
        conti: F,
    ) -> anyhow::Result<&'a Tree<'a>>
    where
        F: for<'c> FnOnce(OperationRef<'a, 'c>) -> anyhow::Result<&'a Tree<'a>>,
    {
        let Some(op) = cursor else {
            return Err(anyhow!("No operation found"));
        };
        if op.name().as_string_ref().as_str()? == "func.func" {
            if let Some(name) = options.target_function.as_deref() {
                let sym_name = op.attribute("sym_name")?;
                debug!("Checking function: {}", sym_name);
                if sym_name.to_string().trim_matches('"') == name {
                    debug!("Found target function: {}", name);
                    return conti(op);
                }
            } else {
                return conti(op);
            }
        }
        locate_function(op.next_in_block(), options, conti)
    }
    fn locate_loop<'a, 'b, F>(
        cursor: Option<OperationRef<'a, 'b>>,
        options: &'_ Options,
        conti: F,
    ) -> anyhow::Result<&'a Tree<'a>>
    where
        F: for<'c> FnOnce(OperationRef<'a, 'c>) -> anyhow::Result<&'a Tree<'a>>,
    {
        let Some(op) = cursor else {
            return Err(anyhow!("No operation found"));
        };
        if op.name().as_string_ref().as_str()? == "affine.for" {
            if let Some(name) = options.target_affine_loop.as_deref() {
                if op.has_attribute(name) {
                    debug!("Found target affine loop: {}", name);
                    return conti(op);
                }
            } else {
                return conti(op);
            }
        }
        locate_loop(op.next_in_block(), options, conti)
    }

    let cursor = body.first_operation();
    locate_function(cursor, options, move |func| {
        let cursor = func
            .region(0)?
            .first_block()
            .ok_or_else(|| anyhow!("function does not have block"))?
            .first_operation();
        if options.start_from_loop || options.target_affine_loop.is_some() {
            return locate_loop(cursor, options, move |for_loop| {
                Ok(context.rcontext().build_tree(for_loop, dom, true)?)
            });
        } else {
            Ok(context.rcontext().build_func_tree(func, dom, true)?)
        }
    })
}

fn main_entry() -> anyhow::Result<()> {
    let options = Options::parse();

    let mut reader = match options.input.as_ref() {
        Some(path) => {
            debug!("Opening input file: {}", path.display());
            Box::new(std::fs::File::open(path)?) as Box<dyn Read>
        }
        None => {
            debug!("Reading from stdin");
            Box::new(std::io::stdin()) as Box<dyn Read>
        }
    };

    let mut writer = match options.output.as_ref() {
        Some(path) => {
            debug!("Opening output file: {}", path.display());
            Box::new(std::fs::File::create(path)?) as Box<dyn std::io::Write>
        }
        None => {
            debug!("Writing to stdout");
            Box::new(std::io::stdout()) as Box<dyn std::io::Write>
        }
    };

    let start_time = std::time::Instant::now();

    match &options.method {
        Method::Barvinok {
            barvinok_arg,
            block_size,
            symbol_lowerbound,
            infinite_repeat,
            num_sets,
            max_operations,
            save_distro,
            parallel_loop_depth,
            threads,
            chunk,
            racetrack_bins,
            nbd_epsilon,
            short_ri_bound,
        } => AnalysisContext::start_with_args(barvinok_arg.as_slice(), |context| {
            let context = &context;
            context.bcontext().set_max_operations(*max_operations);
            let (source, from_c) = obtain_mlir_source(&options, &mut reader)?;
            if from_c {
                context.mcontext().set_allow_unregistered_dialects(true);
            }
            let module = Module::parse(context.mcontext(), &source)
                .ok_or_else(|| anyhow!("Failed to parse module"))?;
            if from_c {
                polygeist::strip_module(&module);
            }
            debug!("Parsed module: {}", module.as_operation());
            let dom = DominanceInfo::new(&module);
            let tree = extract_target(&module, &options, context, &dom)?;
            debug!("Extracted tree: {}", tree);
            let (max_param, _) = utils::get_max_param_ivar(tree)?;
            let mut space = isl::get_timestamp_space((max_param + 1).try_into()?, context, tree)?;
            let local_space: LocalSpace = space.get_space()?.try_into()?;
            for (idx, bound) in symbol_lowerbound.iter().enumerate() {
                let bound = *bound;
                debug!("Setting lower bound for symbol {idx} >= {bound}");
                let constraint = Constraint::new_inequality(local_space.clone())?
                    .set_coefficient_si(barvinok::DimType::Param, idx as i32, 1)?
                    .set_constant_si(-bound)?;
                space = space.add_constraint(constraint)?;
            }
            if *infinite_repeat {
                // Run-twice method: instead of a symbolic repeat count R
                // (whose portions needed L'Hôpital limits), materialize one
                // extra period by prepending a timestamp dim bounded to
                // [0, 2). The second copy (leading dim == 1) observes a full
                // prior period, so its reuse intervals are the steady-state
                // ones, including period-boundary wraparound reuses.
                space = space.insert_dims(barvinok::DimType::Out, 0, 1)?;
                let local_space: LocalSpace = space.get_space()?.try_into()?;
                let lb = Constraint::new_inequality(local_space.clone())?.set_coefficient_si(
                    barvinok::DimType::Out,
                    0,
                    1,
                )?;
                let ub = Constraint::new_inequality(local_space.clone())?
                    .set_coefficient_si(barvinok::DimType::Out, 0, -1)?
                    .set_constant_si(1)?;
                space = space.add_constraint(lb)?.add_constraint(ub)?;
                debug!("space with infinite repeat: {space:?}");
            }
            // Model one loop as `omp parallel for` by materializing the
            // owning thread as timestamp dimension 0. Done here, before
            // `ensure_set_name`, so it follows the same insert-then-name order
            // the run-twice path above uses.
            let parallel_spec = match parallel_loop_depth {
                None => None,
                Some(loop_level) => {
                    if *infinite_repeat {
                        return Err(anyhow!(
                            "--parallel-loop-depth and --infinite-repeat both prepend a timestamp \
                             dimension and have not been validated together"
                        ));
                    }
                    let threads = threads.get();
                    let chunk = match chunk {
                        Some(chunk) => chunk.get(),
                        None => {
                            let trip = parallel::constant_trip_count(tree, *loop_level)
                                .ok_or_else(|| anyhow!(
                                    "the loop at nesting level {loop_level} has symbolic bounds, \
                                     so the default chunk `ceil(trip / threads)` cannot be \
                                     resolved; pass --chunk explicitly"
                                ))?;
                            (trip + i64::from(threads) - 1)
                                .div_euclid(i64::from(threads))
                                .max(1)
                        }
                    };
                    let spec = parallel::ParallelSpec {
                        loop_level: *loop_level,
                        threads,
                        chunk,
                    };
                    let ts_dim = parallel::timestamp_dim_of_loop(tree, spec.loop_level)?;
                    info!(
                        "modeling loop level {} (timestamp dim {ts_dim}) as parallel over {} \
                         thread(s), schedule(static, {})",
                        spec.loop_level, spec.threads, spec.chunk
                    );
                    let (extended, thread_dim) =
                        parallel::add_thread_dim(space, ts_dim, &spec)?;
                    space = extended;
                    Some((spec, thread_dim))
                }
            };
            space = isl::ensure_set_name(space)?;
            let mut access_map = isl::get_access_map(
                (max_param + 1).try_into()?,
                context,
                tree,
                *block_size,
                *num_sets,
            )?;
            if *infinite_repeat {
                // The repeat dim carries no constraints here; intersecting
                // with the timestamp space below bounds it to [0, 2).
                access_map = access_map.insert_dims(barvinok::DimType::In, 0, 1)?;
            }
            if parallel_spec.is_some() {
                // Same story for the thread dim: unconstrained here, pinned by
                // the intersection with the timestamp space below. Appended
                // last, matching where `add_thread_dim` put it.
                let in_dims = access_map.get_space()?.dim(barvinok::DimType::In)?;
                access_map = access_map
                    .insert_dims(barvinok::DimType::In, in_dims, 1)?
                    .set_dim_name(
                        barvinok::DimType::In,
                        in_dims,
                        parallel::THREAD_DIM_NAME,
                    )?;
            }
            let access_map = isl::ensure_map_domain_name(access_map)?;
            let access_map = access_map.intersect_domain(space.clone())?;

            if let Some((spec, thread_dim)) = &parallel_spec {
                let short_bound = match short_ri_bound {
                    Some(bound) => *bound,
                    None => parallel::ShortBound::default().value()?,
                };
                let knobs = parallel::CriKnobs {
                    short_bound,
                    racetrack_bins: racetrack_bins.get(),
                    nbd_epsilon: *nbd_epsilon,
                };
                let report = parallel::analyze(
                    space.clone(),
                    access_map.clone(),
                    *thread_dim,
                    spec,
                    &knobs,
                )?;
                let mut curve = denning::MissRatioCurve::new(&report.support);
                if options.associativity.get() > 1 {
                    curve = curve.compute_assoc(
                        options.associativity.get(),
                        1.0,
                        denning::SkewDecay::Constant,
                    );
                }
                if options.json {
                    let output = serde_json::json!({
                        "parallel": &report,
                        "miss_ratio_curve": &curve,
                        "elapsed_seconds": start_time.elapsed().as_secs_f64(),
                    });
                    writeln!(writer, "{output}")?;
                } else {
                    writeln!(
                        writer,
                        "parallel model: {} thread(s), schedule(static, {}), short-RI bound {:.1}",
                        report.threads, report.chunk, report.short_ri_bound
                    )?;
                    writeln!(
                        writer,
                        "access mass: {:.4} private reuse, {:.4} shared reuse, {:.4} compulsory",
                        report.private_portion, report.shared_portion, report.cold_portion
                    )?;
                    writeln!(
                        writer,
                        "mean sharing degree: {:.2} of {} thread(s)",
                        report.mean_sharing_degree, report.threads
                    )?;
                    writeln!(writer, "CRI support points: {}", report.support.len())?;
                }
                if let Some(path) = &options.miss_ratio_curve {
                    let svgbackend = plotters::backend::SVGBackend::new(
                        path,
                        (
                            options.miss_ratio_curve_width,
                            options.miss_ratio_curve_height,
                        ),
                    );
                    let area = svgbackend.into_drawing_area();
                    curve.plot_miss_ratio_curve(&area)?;
                    info!("Miss ratio curve saved to {}", path.display());
                }
                return Ok(());
            }

            let gt = space.clone().lex_gt_set(space.clone())?;
            let lt = gt.clone().reverse()?;
            let ge = space.clone().lex_ge_set(space.clone())?;
            let access_rev = access_map.clone().reverse()?;
            let same_element = access_map.clone().apply_range(access_rev)?;
            let immediate_pred = same_element.intersect(gt.clone())?.lexmax()?;
            let after = immediate_pred.apply_range(lt)?;
            let ri = after.intersect(ge)?;
            // Under --infinite-repeat keep only reuse intervals whose
            // consuming access (the domain of `ri`) lies in the second copy
            // (leading dim == 1): that copy has a full period of history, so
            // its RIs are the steady-state ones. The reported access total is
            // likewise the cardinality of the second copy only.
            let (ri, total_space) = if *infinite_repeat {
                let local_space: LocalSpace = space.get_space()?.try_into()?;
                let second_copy_eq = Constraint::new_equality(local_space)?
                    .set_coefficient_si(barvinok::DimType::Out, 0, 1)?
                    .set_constant_si(-1)?;
                let second_copy = space.clone().add_constraint(second_copy_eq)?;
                (ri.intersect_domain(second_copy.clone())?, second_copy)
            } else {
                (ri, space.clone())
            };
            let ri_support = ri.clone().domain()?;
            let ri_values = ri.cardinality()?;
            debug!("Timestamp space: {:?}", space);
            debug!("Access map: {:?}", access_map);
            debug!("RI values: {:?}", ri_values);
            let processor = isl::RIProcessor::new(ri_values, ri_support);
            let space_count = total_space.cardinality()?;
            let raw_distro = processor.get_distribution()?;
            if let Some(path) = save_distro {
                isl::save_all_dist_items(&raw_distro, space_count.clone(), path)?;
                info!("Raw distribution saved to {}", path.display());
            }
            if options.json {
                let output = isl::create_json_output(&raw_distro, space_count, start_time)?;
                writeln!(writer, "{output}")?;
            } else {
                let table = isl::create_table(&raw_distro, space_count.clone())?;
                writeln!(writer, "{table}")?;
                writeln!(writer, "Total: {space_count:?}")?;
                match isl::get_distro(&raw_distro, space_count) {
                    Ok(dist) => {
                        let mut curve = denning::MissRatioCurve::new(&dist);
                        // apply associativity only when it's greater than 1
                        if options.associativity.get() > 1 {
                            curve = curve.compute_assoc(
                                options.associativity.get(),
                                1.0,
                                denning::SkewDecay::Constant,
                            );
                        }

                        if let Some(path) = &options.miss_ratio_curve {
                            let svgbackend = plotters::backend::SVGBackend::new(
                                path,
                                (
                                    options.miss_ratio_curve_width,
                                    options.miss_ratio_curve_height,
                                ),
                            );
                            let area = svgbackend.into_drawing_area();
                            curve.plot_miss_ratio_curve(&area)?;
                            info!("Miss ratio curve saved to {}", path.display());
                        }
                    }
                    Err(e) => {
                        error!("Failed to get distribution: {}\n{}", e, e.backtrace());
                    }
                }
            }
            Ok(())
        }),
        Method::Salt { block_size } => AnalysisContext::start(|context| {
            let context = &context;
            let (source, from_c) = obtain_mlir_source(&options, &mut reader)?;
            if from_c {
                context.mcontext().set_allow_unregistered_dialects(true);
            }

            let module = Module::parse(context.mcontext(), &source)
                .ok_or_else(|| anyhow!("Failed to parse module"))?;
            if from_c {
                polygeist::strip_module(&module);
            }

            debug!("Parsed module: {}", module.as_operation());

            let dom = DominanceInfo::new(&module);

            let tree = extract_target(&module, &options, context, &dom)?;

            debug!("Extracted tree: {}", tree);

            // if !is_perfectly_nested(tree) {
            //     return Err(anyhow!("The loop nest is not perfectly nested"));
            // }

            // if !has_reuses(tree) {
            //     return Err(anyhow!(
            //         "The loop nest does not have non-imaginary non-block-wise reuses"
            //     ));
            // }

            // if !no_coefficient_for_block(tree) {
            //     return Err(anyhow!(
            //         "The loop nest has non-one coefficient for block induction variable"
            //     ));
            // }

            let access_cnt = salt::number_of_accesses(tree);

            //  utils::walk_tree_print_converted_affine_map(tree, 0)?;
            let mut rf = HashMap::new();
            let mut tc = HashMap::new();
            let ri_dist = salt::get_reuse_interval_distribution(tree, &mut rf, &mut tc, 1, context);
            // hashmap to vector tuple
            let ri_dist_vec = ri_dist.iter().map(|(k, v)| (k.clone(), v.clone()));

            let ri_dist_vec = match block_size {
                Some(block_size) => ri_dist_vec
                    .map(|(x, y)| {
                        let x = salt::subsitute_block_size(&x, *block_size);
                        let y = salt::subsitute_block_size(&y, *block_size);
                        (x, y)
                    })
                    .collect::<Vec<_>>(),
                None => ri_dist_vec.collect::<Vec<_>>(),
            };
            if options.json {
                let output =
                    salt::create_json_output(&ri_dist_vec, access_cnt, tc.values(), start_time)?;
                writeln!(writer, "{output}")?;
            } else {
                let table = create_table(&ri_dist_vec);
                writeln!(writer, "{table}")?;
                let total_count = salt::get_total_count(access_cnt, tc.values())?;
                writeln!(writer, "Total: {total_count}")?;
                match salt::get_ri_distro(&ri_dist_vec) {
                    Ok(dist) => {
                        let mut curve = denning::MissRatioCurve::new(&dist);
                        // apply associativity only when it's greater than 1
                        if options.associativity.get() > 1 {
                            curve = curve.compute_assoc(
                                options.associativity.get(),
                                1.0,
                                denning::SkewDecay::Constant,
                            );
                        }

                        if let Some(path) = &options.miss_ratio_curve {
                            let svgbackend = plotters::backend::SVGBackend::new(
                                path,
                                (
                                    options.miss_ratio_curve_width,
                                    options.miss_ratio_curve_height,
                                ),
                            );
                            let area = svgbackend.into_drawing_area();
                            curve.plot_miss_ratio_curve(&area)?;
                            info!("Miss ratio curve saved to {}", path.display());
                        }
                    }
                    Err(e) => {
                        error!("Failed to get distribution: {}\n{}", e, e.backtrace());
                    }
                }
            }
            Ok(())
        }),
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let res = main_entry();
    if let Err(e) = res {
        let bt = e.backtrace();
        error!("\n{:#}", e);
        match bt.status() {
            std::backtrace::BacktraceStatus::Disabled => {
                error!("Backtrace is disabled -- rerun with RUST_BACKTRACE=1 to get a backtrace")
            }
            std::backtrace::BacktraceStatus::Captured => error!("Stack backtrace:\n{bt}"),
            _ => panic!("unsupported backtrace"),
        }
        std::process::exit(1);
    }
}
