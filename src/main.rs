/*
    Copyright 2023 - Raphaël Rigo

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/
// Includes (many) changes by Valentin Obst.

mod corpus;
mod output;
mod plotting;

use crate::corpus::{is_strict, load_corpus, CorpusStats};
use crate::output::CliJsonOutput;
use crate::plotting::PlotOptions;

use std::cmp::min;
use std::collections::{BTreeMap, HashMap};
use std::convert::From;
use std::io;
use std::ops::Range;

use anyhow::{Context, Result};
use clap::{arg, Arg, ArgAction};
use log::{debug, info};
use rayon::prelude::*;

#[derive(Debug)]
struct KlRes {
    arch: String,
    div: f64,
}

struct RangeFullKlRes {
    kl_bg: Vec<KlRes>,
    kl_tg: Vec<KlRes>,
}

fn calculate_kl(corpus_stats: &[CorpusStats], target: &CorpusStats) -> RangeFullKlRes {
    let mut kl_bg = Vec::<KlRes>::with_capacity(corpus_stats.len());
    let mut kl_tg = Vec::<KlRes>::with_capacity(corpus_stats.len());

    for arch_stats in corpus_stats {
        let r = target.compute_kl(arch_stats);
        kl_bg.push(KlRes {
            arch: arch_stats.arch.clone(),
            div: r.bigrams,
        });
        kl_tg.push(KlRes {
            arch: arch_stats.arch.clone(),
            div: r.trigrams,
        });
    }

    // Sort
    kl_bg.sort_unstable_by(|a, b| a.div.partial_cmp(&b.div).unwrap());
    debug!("Results 2-gram: {:?}", &kl_bg[0..2]);
    kl_tg.sort_unstable_by(|a, b| a.div.partial_cmp(&b.div).unwrap());
    debug!("Results 3-gram: {:?}", &kl_tg[0..2]);

    RangeFullKlRes { kl_bg, kl_tg }
}

struct ProcessedDetectionResult {
    pub win_sz: usize,
    pub max_kl_bg: f64,
    pub min_kl_bg: f64,
    pub max_kl_tg: f64,
    pub min_kl_tg: f64,
    pub range_to_result_bg: HashMap<Range<usize>, RangeResult>,
    pub range_to_result_tg: HashMap<Range<usize>, RangeResult>,
    pub arch_to_idx: HashMap<Arch, usize>,
    pub idx_to_arch: HashMap<usize, Arch>,
    pub kl_arch_to_range_bg: BTreeMap<Arch, Vec<(Range<usize>, f64)>>,
    pub kl_arch_to_range_tg: BTreeMap<Arch, Vec<(Range<usize>, f64)>>,
    pub range_to_final_result: HashMap<Range<usize>, Option<Arch>>,
    pub arch_to_final_ranges: HashMap<Arch, Vec<Range<usize>>>,
}

pub struct RangeResult {
    arch: Arch,
    div: f64,
    range_mean: f64,
    range_var: f64,
}

/// Main heuristic that decides which arch is assigned to a range.
pub fn final_range_result(res_bg: &RangeResult, res_tg: &RangeResult) -> Option<Arch> {
    let RangeResult {
        arch: arch_bg,
        div: div_bg,
        range_mean: mean_bg,
        range_var: var_bg,
    } = res_bg;
    let std_deviation_bg = var_bg.sqrt();
    let RangeResult {
        arch: arch_tg,
        div: div_tg,
        range_mean: mean_tg,
        range_var: var_tg,
    } = res_tg;
    let std_deviation_tg = var_tg.sqrt();

    // Limits on the absolute divergence of the closest arch.
    const MAX_ABS_DIV_BG: f64 = 5.0;
    const MAX_ABS_DIV_TG: f64 = 6.0;
    const MAX_ABS_DIV_STRICT_BG: f64 = 4.0;
    const MAX_ABS_DIV_STRICT_TG: f64 = 5.0;

    // Threshold for instant detection via standard deviation.
    const INSTANT_STD_DEV_BG: f64 = 2.0;
    const INSTANT_STD_DEV_TG: f64 = 2.0;
    const INSTANT_STD_DEV_STRICT_BG: f64 = 2.5;
    const INSTANT_STD_DEV_STRICT_TG: f64 = 2.5;

    // Threshold for conditional detection via standard deviation.
    const COMM_STD_DEV_BG: f64 = 1.0;
    const COMM_STD_DEV_TG: f64 = 1.0;
    const COMM_STD_DEV_STRICT_BG: f64 = 1.5;
    const COMM_STD_DEV_STRICT_TG: f64 = 1.5;

    let (max_abs_div_bg, instant_std_dev_bg, comm_std_dev_bg): (f64, f64, f64) =
        if is_strict(arch_bg) {
            (
                MAX_ABS_DIV_STRICT_BG,
                INSTANT_STD_DEV_STRICT_BG,
                COMM_STD_DEV_STRICT_BG,
            )
        } else {
            (MAX_ABS_DIV_BG, INSTANT_STD_DEV_BG, COMM_STD_DEV_BG)
        };
    let (max_abs_div_tg, instant_std_dev_tg, comm_std_dev_tg): (f64, f64, f64) =
        if is_strict(arch_tg) {
            (
                MAX_ABS_DIV_STRICT_TG,
                INSTANT_STD_DEV_STRICT_TG,
                COMM_STD_DEV_STRICT_TG,
            )
        } else {
            (MAX_ABS_DIV_TG, INSTANT_STD_DEV_TG, COMM_STD_DEV_TG)
        };

    #[allow(clippy::if_same_then_else)]
    // Detect nothing if the closest arch is too far away in absolute numbers.
    if div_bg.partial_cmp(&max_abs_div_bg).unwrap() == core::cmp::Ordering::Greater
        && div_tg.partial_cmp(&max_abs_div_tg).unwrap() == core::cmp::Ordering::Greater
    {
        None
    // Instant detection if an arch is clearly the best in either tri- or
    // bigrams. Test trigrams first as they seem to be somewhat better.
    } else if div_tg
        .partial_cmp(&(mean_tg - instant_std_dev_tg * std_deviation_tg))
        .unwrap()
        == core::cmp::Ordering::Less
    {
        Some(arch_tg.clone())
    } else if div_bg
        .partial_cmp(&(mean_bg - instant_std_dev_bg * std_deviation_bg))
        .unwrap()
        == core::cmp::Ordering::Less
    {
        Some(arch_bg.clone())
    // Main heuristic: Bi- and trigrams agree and the divergence stands out from
    // the others.
    } else if div_bg
        .partial_cmp(&(mean_bg - comm_std_dev_bg * std_deviation_bg))
        .unwrap()
        == core::cmp::Ordering::Less
        && div_tg
            .partial_cmp(&(mean_tg - comm_std_dev_tg * std_deviation_tg))
            .unwrap()
            == core::cmp::Ordering::Less
        && arch_tg == arch_bg
    {
        Some(arch_tg.clone())
    // Special case for detection of text via trigrams.
    } else if div_tg
        .partial_cmp(&(mean_tg - 1.0 * std_deviation_tg))
        .unwrap()
        == core::cmp::Ordering::Less
        && arch_tg.starts_with("_words")
    {
        Some(arch_tg.clone())
    } else {
        None
    }
}

impl From<(Arch, f64, f64, f64)> for RangeResult {
    fn from(i: (Arch, f64, f64, f64)) -> Self {
        Self {
            arch: i.0,
            div: i.1,
            range_mean: i.2,
            range_var: i.3,
        }
    }
}

pub fn calculate_mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / (data.len() as f64)
}

pub fn calculate_variance(data: &[f64], mean: f64) -> f64 {
    data.iter().map(|x| f64::powi(x - mean, 2)).sum::<f64>() / (data.len() as f64)
}

impl From<DetectionResult> for ProcessedDetectionResult {
    fn from(res_ex: DetectionResult) -> Self {
        // Size of a range.
        let win_sz = res_ex.kl_bg_range_to_arch.keys().next().unwrap().len();

        // Numbering of arches.
        let mut arch_to_idx: HashMap<Arch, usize> = HashMap::new();
        let mut idx_to_arch: HashMap<usize, Arch> = HashMap::new();
        for (arch_idx, (arch, _res)) in res_ex.kl_bg_arch_to_range.iter().enumerate() {
            arch_to_idx.insert(arch.clone(), arch_idx);
            idx_to_arch.insert(arch_idx, arch.clone());
        }

        // Global max and min.
        let mut all_divs_bg: Vec<f64> = res_ex
            .kl_bg_arch_to_range
            .values()
            .flat_map(|arch| arch.iter().map(|(_, div)| *div))
            .collect();
        all_divs_bg.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let max_kl_bg = *all_divs_bg.last().unwrap();
        let min_kl_bg = *all_divs_bg
            .iter()
            .find(|div| (*div).partial_cmp(&0.1).unwrap() != core::cmp::Ordering::Less)
            .unwrap();
        let mut all_divs_tg: Vec<f64> = res_ex
            .kl_tg_arch_to_range
            .values()
            .flat_map(|arch| arch.iter().map(|(_, div)| *div))
            .collect();
        all_divs_tg.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let max_kl_tg = *all_divs_tg.last().unwrap();
        let min_kl_tg = *all_divs_tg
            .iter()
            .find(|div| (*div).partial_cmp(&0.1).unwrap() != core::cmp::Ordering::Less)
            .unwrap();

        // Per-range min (with arch), mean, and variance.
        let range_to_result_bg: HashMap<Range<usize>, RangeResult> = res_ex
            .kl_bg_range_to_arch
            .iter()
            .map(|(range, arches)| {
                let mut arches = arches.clone();
                arches.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                let divs: Vec<_> = arches.iter().map(|(_, div)| *div).collect();

                let mean = calculate_mean(&divs);
                let var = calculate_variance(&divs, mean);

                (
                    range.clone(),
                    (arches[0].0.clone(), arches[0].1, mean, var).into(),
                )
            })
            .collect();
        let range_to_result_tg: HashMap<Range<usize>, RangeResult> = res_ex
            .kl_tg_range_to_arch
            .iter()
            .map(|(range, arches)| {
                let mut arches = arches.clone();
                arches.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                let divs: Vec<_> = arches.iter().map(|(_, div)| *div).collect();

                let mean = calculate_mean(&divs);
                let var = calculate_variance(&divs, mean);

                (
                    range.clone(),
                    (arches[0].0.clone(), arches[0].1, mean, var).into(),
                )
            })
            .collect();

        // Our final verdict.
        let range_to_final_result: HashMap<Range<usize>, Option<String>> = range_to_result_bg
            .iter()
            .map(|(range, res_bg)| {
                let res_tg = range_to_result_tg.get(range).unwrap();

                (range.clone(), final_range_result(res_bg, res_tg))
            })
            .collect();

        let mut arch_to_final_ranges: HashMap<Arch, Vec<Range<usize>>> = HashMap::new();
        for (range, arch_op) in range_to_final_result.iter() {
            if let Some(arch) = arch_op {
                arch_to_final_ranges
                    .entry(arch.clone())
                    .and_modify(|ranges| ranges.push(range.clone()))
                    .or_insert(vec![range.clone()]);
            }
        }

        Self {
            win_sz,
            arch_to_idx,
            idx_to_arch,
            max_kl_bg,
            min_kl_bg,
            max_kl_tg,
            min_kl_tg,
            range_to_result_bg,
            range_to_result_tg,
            kl_arch_to_range_bg: res_ex.kl_bg_arch_to_range,
            kl_arch_to_range_tg: res_ex.kl_tg_arch_to_range,
            range_to_final_result,
            arch_to_final_ranges,
        }
    }
}

type Arch = String;
struct DetectionResult {
    pub kl_bg_arch_to_range: BTreeMap<Arch, Vec<(Range<usize>, f64)>>,
    pub kl_tg_arch_to_range: BTreeMap<Arch, Vec<(Range<usize>, f64)>>,
    pub kl_bg_range_to_arch: HashMap<Range<usize>, Vec<(Arch, f64)>>,
    pub kl_tg_range_to_arch: HashMap<Range<usize>, Vec<(Arch, f64)>>,
}

impl<I: ParallelIterator<Item = (Range<usize>, RangeFullKlRes)>> From<I> for DetectionResult {
    fn from(i: I) -> Self {
        let mut res_ex = Self {
            kl_bg_arch_to_range: BTreeMap::new(),
            kl_tg_arch_to_range: BTreeMap::new(),
            kl_bg_range_to_arch: HashMap::new(),
            kl_tg_range_to_arch: HashMap::new(),
        };
        let res: Vec<_> = i.collect();

        for (range, RangeFullKlRes { kl_bg, kl_tg }) in res {
            for (kl_bg_arch, kl_tg_arch) in kl_bg.into_iter().zip(kl_tg.into_iter()) {
                res_ex
                    .kl_bg_arch_to_range
                    .entry(kl_bg_arch.arch.clone())
                    .and_modify(|e| e.push((range.clone(), kl_bg_arch.div)))
                    .or_insert(vec![(range.clone(), kl_bg_arch.div)]);
                res_ex
                    .kl_tg_arch_to_range
                    .entry(kl_tg_arch.arch.clone())
                    .and_modify(|e| e.push((range.clone(), kl_tg_arch.div)))
                    .or_insert(vec![(range.clone(), kl_tg_arch.div)]);
                res_ex
                    .kl_bg_range_to_arch
                    .entry(range.clone())
                    .and_modify(|e| e.push((kl_bg_arch.arch.clone(), kl_bg_arch.div)))
                    .or_insert(vec![(kl_bg_arch.arch, kl_bg_arch.div)]);
                res_ex
                    .kl_tg_range_to_arch
                    .entry(range.clone())
                    .and_modify(|e| e.push((kl_tg_arch.arch.clone(), kl_tg_arch.div)))
                    .or_insert(vec![(kl_tg_arch.arch.clone(), kl_tg_arch.div)]);
            }
        }

        res_ex
    }
}

fn detect_code(corpus_stats: &[CorpusStats], file_data: &[u8], filename: &str) -> DetectionResult {
    // Heuristic depending on file size, the number is actually half the window
    // size.
    let window = match file_data.len() {
        0x100001..=0x1000000 => 0x1000, // 257 - 4096, 1MiB - 16MiB
        0x20001..=0x100000 => 0x800,    // 65 - 512, 128KiB - 1MiB
        0x8001..=0x20000 => 0x400,      // 33 - 128, 32KiB - 128KiB
        0x1001..=0x8000 => 0x200,       // 9 - 64, 4KiB - 32KiB
        0..=0x1000 => 0x100,            // 1 - 16, 0B - 4KiB
        // From here on we grow the number of windows logarithmically in the
        // file size. Constant factor ensures smooth transition.
        l => (l / (170 * ((l as f64).log2() as usize))) & 0xFFFFF000,
    };

    info!("{}: window_size : 0x{:x} ", filename, window * 2);

    let res_ex: DetectionResult = (0..file_data.len())
        .into_par_iter()
        .step_by(window * 2)
        .map(|start| {
            let end = min(file_data.len(), start + window * 2);

            let win_stats = CorpusStats::new("target".to_string(), &file_data[start..end], 0.0);

            let range_res = calculate_kl(corpus_stats, &win_stats);

            (start..end, range_res)
        })
        .into();

    res_ex
}

fn hex_to_int(arg: &str) -> Result<u64, std::num::ParseIntError> {
    let tmp = arg.trim_start_matches("0x");
    u64::from_str_radix(tmp, 16)
}

fn main() -> Result<()> {
    let app = clap::Command::new("coderec")
        .version(env!("CARGO_PKG_VERSION"))
        .propagate_version(true)
        .author("Valentin Obst <coderec@vpao.io>")
        .about("Identifies machine code in binary files.")
        .arg(arg!(-d - -debug))
        .arg(arg!(-q - -quiet))
        .arg(arg!(-v - -verbose))
        .arg(arg!(-B --"big-file" "Optimized analysis for files larger than X00MiB."))
        .arg(arg!(--"plot-corpus" "Plot distributions of samples in corpus and exit."))
        .arg(arg!(--"plot-divs" "Plot raw analysis results in addition to region plot."))
        .arg(arg!(--"no-plots" "Do not generate any plots."))
        .arg(arg!(--"no-out" "Do not write detection results to stdout."))
        .arg(arg!(--"si-labels" "Use decimal SI-prefix X-axis labels (e.g. 1M) instead of hex."))
        .arg(arg!(--"no-title" "Do not add a title to the generated plot."))
        .arg(
            Arg::new("offset")
                .short('o')
                .long("offset")
                .required(false)
                .requires("length")
                .action(clap::ArgAction::Set)
                .value_parser(hex_to_int)
                .help("Offset into the file where analysis starts."),
        )
        .arg(
            Arg::new("length")
                .short('l')
                .long("length")
                .required(false)
                .action(clap::ArgAction::Set)
                .value_parser(hex_to_int)
                .help("Number of bytes that are analyzed."),
        )
        .arg(
            Arg::new("base")
                .short('b')
                .long("base-address")
                .required(false)
                .action(clap::ArgAction::Set)
                .value_parser(hex_to_int)
                .help("Base address of the file.")
                .default_value("0"),
        )
        .arg(
            Arg::new("output")
                .short('O')
                .long("output-path")
                .required(false)
                .action(clap::ArgAction::Set)
                .help("Output path for the generated images."),
        )
        .arg(
            Arg::new("format")
                .short('F')
                .long("format")
                .required(false)
                .action(clap::ArgAction::Set)
                .value_parser(clap::builder::PossibleValuesParser::new(["png", "svg"]))
                .help("Output format for images.")
                .default_value("png"),
        )
        .arg(
            Arg::new("files")
                .action(ArgAction::Append)
                .value_parser(clap::builder::NonEmptyStringValueParser::new())
                .required_unless_present("plot-corpus"),
        );

    let args = app.get_matches();

    let level = if args.get_flag("debug") {
        log::Level::Debug
    } else if args.get_flag("verbose") {
        log::Level::Info
    } else if args.get_flag("quiet") {
        log::Level::Error
    } else {
        log::Level::Warn
    };
    simple_logger::init_with_level(level)?;

    let plot_opts = PlotOptions {
        big_file: args.get_flag("big-file"),
        si_labels: args.get_flag("si-labels"),
        no_title: args.get_flag("no-title"),
        output_path: args.get_one::<String>("output").cloned(),
        output_format: args.get_one::<String>("format").unwrap().clone(),
    };

    let base_address: u64 = *args.get_one("base").unwrap();

    let corpus_stats = load_corpus();

    if args.get_flag("plot-corpus") {
        for arch in corpus_stats.iter() {
            arch.plot_tg();
            arch.plot_cond_prob();
        }

        return Ok(());
    }

    info!("Corpus size: {}", corpus_stats.len());

    for file in args.get_many::<String>("files").unwrap() {
        let file_data = std::fs::read(file).with_context(|| format!("Could not open {}", file))?;

        let (data, name, base_address) = if let Some(offset) = args.get_one::<u64>("offset") {
            let length: &u64 = args.get_one("length").unwrap();
            let name = format!("{}_o{:x}_l{:x}", file, offset, length);

            (
                &file_data[*offset as usize..(offset + length) as usize],
                name,
                base_address + *offset,
            )
        } else {
            (file_data.as_slice(), file.clone(), base_address)
        };

        let raw_res = detect_code(&corpus_stats, data, &name);
        let processes_res: ProcessedDetectionResult = raw_res.into();

        if !args.get_flag("no-plots") {
            if args.get_flag("plot-divs") {
                crate::plotting::plot_divs(&name, data.len(), &processes_res);
            }

            crate::plotting::plot_regions(
                &name,
                data.len(),
                data,
                &processes_res,
                base_address,
                &plot_opts,
            );
        }

        if !args.get_flag("no-out") {
            serde_json::to_writer(
                io::stdout().lock(),
                &CliJsonOutput::from((name.as_str(), &processes_res)),
            )
            .unwrap()
        }
    }

    Ok(())
}
