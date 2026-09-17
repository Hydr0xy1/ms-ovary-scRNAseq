#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Matrix)
  library(SeuratObject)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: 15_extract_gse232309.R INPUT_RDS_GZ POPULATION OUTPUT_DIR")
}
input <- args[[1]]
population <- args[[2]]
output_dir <- args[[3]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# GEO stores these files as an outer .gz archive around an already
# gzip-compressed RDS stream.  A second gzip connection is therefore needed;
# reading only gzfile(input) leaves the inner gzip header for readRDS and
# produces "unknown input format".
input_connection <- gzcon(gzfile(input, open = "rb"))
on.exit(close(input_connection), add = TRUE)
obj <- readRDS(input_connection)
metadata <- obj[[]]
if (nrow(metadata) == 0) stop("Seurat metadata is empty")

pick_column <- function(candidates, columns) {
  found <- candidates[candidates %in% columns]
  if (length(found) == 0) return(NA_character_)
  found[[1]]
}

sample_col <- pick_column(
  c("sample_id", "sample", "Sample", "SampleID", "orig.ident", "replicate", "Replicate"),
  colnames(metadata)
)
age_col <- pick_column(
  c("age_group", "age", "Age", "group", "Group", "condition", "Condition", "stage", "Stage"),
  colnames(metadata)
)
if (is.na(sample_col)) {
  stop(paste("No biological sample column. Metadata columns:", paste(colnames(metadata), collapse = ",")))
}

sample_id <- as.character(metadata[[sample_col]])
if (is.na(age_col)) {
  age_raw <- sample_id
} else {
  age_raw <- as.character(metadata[[age_col]])
}
young_pattern <- "(^|[^0-9])(3|03)(m|mo|month)|young|^y($|[_-])"
aged_pattern <- "(^|[^0-9])(9|09)(m|mo|month)|old|aged|^o($|[_-])"
age_group <- ifelse(
  grepl(young_pattern, age_raw, ignore.case = TRUE, perl = TRUE), "Young",
  ifelse(grepl(aged_pattern, age_raw, ignore.case = TRUE, perl = TRUE), "Aged", NA_character_)
)
if (anyNA(age_group)) {
  stop(paste(
    "Could not map all age labels. age column:", ifelse(is.na(age_col), "sample_id", age_col),
    "values:", paste(sort(unique(age_raw)), collapse = ",")
  ))
}

assay <- DefaultAssay(obj)
counts <- tryCatch(
  GetAssayData(obj, assay = assay, layer = "counts"),
  error = function(e) GetAssayData(obj, assay = assay, slot = "counts")
)
if (!inherits(counts, "sparseMatrix")) counts <- as(counts, "dgCMatrix")
if (ncol(counts) != length(sample_id)) stop("Counts and metadata cell counts differ")

sample_levels <- sort(unique(sample_id))
design <- sparse.model.matrix(~ 0 + factor(sample_id, levels = sample_levels))
colnames(design) <- sample_levels
pseudobulk <- counts %*% design

sample_metadata <- data.frame(
  sample_id = sample_levels,
  age_group = vapply(sample_levels, function(x) unique(age_group[sample_id == x])[[1]], character(1)),
  population = population,
  n_cells = vapply(sample_levels, function(x) sum(sample_id == x), integer(1)),
  total_umi = as.numeric(colSums(pseudobulk)),
  source_sample_column = sample_col,
  source_age_column = ifelse(is.na(age_col), "derived_from_sample_id", age_col),
  stringsAsFactors = FALSE
)
if (any(vapply(sample_levels, function(x) length(unique(age_group[sample_id == x])) != 1, logical(1)))) {
  stop("A biological sample maps to more than one age group")
}

output_counts <- as.data.frame(as.matrix(t(pseudobulk)))
output_counts <- cbind(sample_id = rownames(output_counts), output_counts)
write.table(
  output_counts,
  gzfile(file.path(output_dir, paste0(population, "__pseudobulk_counts.tsv.gz"))),
  sep = "\t", row.names = FALSE, quote = FALSE
)
write.table(
  sample_metadata,
  file.path(output_dir, paste0(population, "__sample_metadata.tsv")),
  sep = "\t", row.names = FALSE, quote = FALSE
)
write.table(
  data.frame(metadata_column = colnames(metadata)),
  file.path(output_dir, paste0(population, "__metadata_columns.tsv")),
  sep = "\t", row.names = FALSE, quote = FALSE
)
cat(sprintf("EXTERNAL_PSEUDOBULK_COMPLETE population=%s samples=%d genes=%d\n", population, nrow(output_counts), nrow(counts)))
