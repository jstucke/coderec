/*
    Copyright 2024 - Valentin Obst <coderec@vpao.io>

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
//! Command line JSON output.

use crate::{Arch, ProcessedDetectionResult};

use std::convert::From;
use std::ops::Range;

use serde::Serialize;

/// Information that is printed to stdout for each analyzed file.
#[derive(Serialize)]
pub struct CliJsonOutput {
    /// Name of the analyzed file.
    file: String,
    /// Consolidated detection results.
    range_results: Vec<(Range<usize>, usize, Arch)>,
}

impl From<(&str, &ProcessedDetectionResult)> for CliJsonOutput {
    fn from((file, res): (&str, &ProcessedDetectionResult)) -> Self {
        let mut range_to_final_result: Vec<_> = res
            .range_to_final_result
            .iter()
            .filter(|(_, arch_op)| arch_op.is_some())
            .collect();
        range_to_final_result.sort_unstable_by(|(a, _), (b, _)| a.start.cmp(&b.start));

        let range_results = range_to_final_result
            .into_iter()
            .fold(
                Vec::<(Range<usize>, Arch)>::new(),
                |mut acc, (range, arch_op)| {
                    let arch = arch_op.as_ref().unwrap();
                    if let Some((last_range, last_arch)) = acc.last_mut() {
                        // merge if same type and adjacent/overlapping
                        if last_arch == arch && range.start <= last_range.end {
                            last_range.end = last_range.end.max(range.end);
                            return acc;
                        }
                        // if ranges overlap: trim end to start of next range
                        if range.start < last_range.end {
                            last_range.end = range.start;
                        }
                    }
                    acc.push((range.clone(), arch.clone()));
                    acc
                },
            )
            .into_iter()
            .map(|(range, arch)| {
                let len = range.end - range.start;
                (range, len, arch)
            })
            .collect();

        CliJsonOutput {
            file: file.to_owned(),
            range_results,
        }
    }
}
