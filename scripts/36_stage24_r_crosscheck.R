#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop(
    "Usage: 36_stage24_r_crosscheck.R counts.tsv.gz output.tsv.gz target_gene n_cores"
  )
}

counts_path <- args[[1]]
output_path <- args[[2]]
target_gene <- args[[3]]
n_cores <- as.integer(args[[4]])

suppressPackageStartupMessages({
  library(scTenifoldKnk)
})

counts <- as.matrix(
  read.delim(
    gzfile(counts_path),
    row.names = 1,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
)
storage.mode(counts) <- "double"
if (!target_gene %in% rownames(counts)) {
  stop("Target gene is absent from the cross-check matrix: ", target_gene)
}

# The official function performs CPM normalization internally. QC thresholds
# are neutralized because cell and gene eligibility were frozen upstream.
result <- scTenifoldKnk(
  countMatrix = counts,
  gKO = target_gene,
  transcriptomeWide = TRUE,
  qc = TRUE,
  qc_minLibSize = 0,
  qc_removeOutlierCells = FALSE,
  qc_minPCT = 0,
  qc_maxMTratio = 1,
  nc_lambda = 0,
  nc_nNet = 10,
  nc_nCells = min(500L, ncol(counts)),
  nc_nComp = 3,
  nc_scaleScores = TRUE,
  nc_symmetric = FALSE,
  nc_q = 0.9,
  td_K = 3,
  td_maxIter = 1000,
  td_maxError = 1e-5,
  td_nDecimal = 3,
  ma_nDim = 2,
  dr_empiricalNull = FALSE,
  nCores = n_cores
)

distance <- as.numeric(result$perturbationDistances[target_gene, ])
output <- data.frame(
  gene = colnames(result$perturbationDistances),
  distance = distance,
  stringsAsFactors = FALSE
)
output <- output[order(-output$distance, output$gene), , drop = FALSE]
output$rank <- seq_len(nrow(output))
output$rank_fraction <- output$rank / nrow(output)

con <- gzfile(output_path, open = "wt")
write.table(output, con, sep = "\t", quote = FALSE, row.names = FALSE)
close(con)

writeLines(capture.output(sessionInfo()), paste0(output_path, ".sessionInfo.txt"))
