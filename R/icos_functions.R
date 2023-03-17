## =============================================================================
##
## icos_functions.R
##
## This script is used for loading .gpkg files resulting from preprocessing in
## prisma_ggp_preproc_main.py. The .gpkgs contain combined information about
## estimated GPP at different ICOS sites (and metadata), associated cropped and
## averaged hyperspectral bands as well as PPI (derived from Copernicus S2 data)
## and SPEI (derived from E-OBS data).
## 
## Author: Floris Hermanns
##
## Date Created: 2023-03-17
##
## =============================================================================

load_flx_gdf <- function(wdir, sensor, sr, mask, rad, aggr = 'na',
                         response = 'GPP_DT_VUT_50', dr = FALSE, return_df = FALSE,
                         forest_only = FALSE, site_only = NULL, year_only = NULL) {
  #' @title Load ICOS + HSI geopackage
  #'
  #' @description Load a geopackage file of preprocessed & cropped hyperspectral
  #' information combined with time-matched flux tower data for analyses like
  #' dimension reduction followed by a (supervised) statistical learning approach.
  #' 
  #' @param wdir directory containing .gpkg files of flux and hyperspectral data
  #' @param sensor type of hyperspectral sensor, either "PRISMA" or "DESIS".
  #' @param sr spectral range of the input data, either "vnir" (visible & near
  #' infrared) or "vswir" (visible, near- & shortwave infrared).
  #' @param mask type of spatial statistic, either "ffp" (flux footprint) or
  #' "zon" (zonal statistic).
  #' @param rad type of spectral data, either "upw" (upwelling radiation) or
  #' "ref" (reflectance).
  #' @param aggr The type of value aggregation that has been performed. Either
  #' "na" (no aggr.), "sum" (daily sum) or "mean" (daily daytime mean).
  #' @param response The response variable to be included in the output data.frame.
  #' Can be any ecosystem productivity variable measured at ICOS sites.
  #' @param dr If dimension reduction (e.g. PCA) has been applied before,
  #' different column names will be used.
  #' @param return_df If true, a regular data frame instead of a sf spatial data
  #' frame will be returned.
  #' @param forest_only If true, only ICOS sites with the IGBP classes 'MF',
  #' 'DBF', 'DNF', 'ENF' and 'EBF' will be included.
  #' @param site_only Use the ICOS abbreviation of a specific site as argument
  #' to return only observations from this site.
  #' @param year_only Only return observations from the selected year.
  #' 
  #' @return a list containing the gap-filled, cleaned geopackage as a data.frame
  #' and the number of hyperspectral bands.
  #' 
  
  fname <- ifelse(dr == F, sprintf('all_sites_%s_%s_%s_%s_%s_covars.gpkg', sensor, sr, mask, rad, aggr),
                  sprintf('all_sites_%s_sivm_%s_%s_%s.gpkg', sensor, sr, mask, rad))
  #fname <- sprintf('all_sites_%s_%s_%s_%s_covars.gpkg', sensor, sr, mask, rad)
  fpath <- file.path(wdir, 'out', fname)
  
  flx_db <- st_transform(st_read(file.path(wdir, 'flx_sites.gpkg')), 3035)
  flx_db <- cbind(flx_db, st_coordinates(flx_db))
  
  flx_hsi_gdf <- st_read(fpath)
  flx_hsi_gdf$ecosystem <- 0
  # import ecosystem type from flx_sites file
  for (site in unique(flx_hsi_gdf$name)) {
    flx_hsi_gdf[flx_hsi_gdf$name == site, 'ecosystem'] <- flx_db[flx_db$name == site, 'ecosystem', drop=TRUE]
    flx_hsi_gdf[flx_hsi_gdf$name == site, 'X'] <- flx_db[flx_db$name == site, 'X', drop=TRUE]
    flx_hsi_gdf[flx_hsi_gdf$name == site, 'Y'] <- flx_db[flx_db$name == site, 'Y', drop=TRUE]
  }
  if (forest_only == TRUE) {
    flx_hsi_gdf = flx_hsi_gdf[flx_hsi_gdf$ecosystem  %in% c('ENF', 'EBF', 'DBF', 'MF'), ]
  }
  
  ppi_z_ix = which(flx_hsi_gdf$PPI == 0)
  if (length(ppi_z_ix) != 0L) {
    zero_sites = flx_hsi_gdf[ppi_z_ix, 'name', drop=TRUE]
    message(sprintf('rows %s (%s) contain zero PPI values and will be removed.',
                    paste(sprintf('%s', ppi_z_ix), collapse=','),
                    paste(sprintf('%s', zero_sites), collapse=',')))
    flx_hsi_gdf <- flx_hsi_gdf[-ppi_z_ix, ]
  }
  gpp_z_ix = which(flx_hsi_gdf[, response, drop=TRUE] == 0)
  if (length(gpp_z_ix) != 0L) {
    zero_sites = flx_hsi_gdf[gpp_z_ix, 'name', drop=TRUE]
    message(sprintf('rows %s (%s) contain zero productivity values and will be removed.',
                    paste(sprintf('%s', gpp_z_ix), collapse=','),
                    paste(sprintf('%s', zero_sites), collapse=',')))
    flx_hsi_gdf <- flx_hsi_gdf[-gpp_z_ix, ]
  }
  rownames(flx_hsi_gdf) <- NULL # reset index
  
  cn <- colnames(flx_hsi_gdf)
  last_band <- ifelse(dr == FALSE, max(cn[grepl('b', cn)]), max(cn[grepl('comp', cn)])) # max returns alphanumerically last value
  nbands <- ifelse(dr == FALSE, as.integer(substr(last_band, 2, 4)), as.integer(substr(last_band, 5, 6)))
  
  # check for NAs in hyperspectral data
  if (dr == FALSE) {
    X <- flx_hsi_gdf[, sprintf('b%03d', seq(1:nbands)), drop=TRUE]
    na_cols <- unique(which(is.na(X), arr.ind = T)[ ,2])
    if (length(na_cols) != 0) {
      message(sprintf('%s: bands %s contain NAs. Will be gapfilled with mean from adjacent bands.',
                      fname, paste(sprintf('%s', na_cols), collapse=',')))
      for (col in na_cols) {
        na_ix <- which(is.na(X[col]))
        if (length(na_ix) > 1) {
          for (ix in na_ix) {
            X[ix, col] <- mean(X[ix-1, col], X[ix+1, col])
          }
        } else if (length(na_ix) == 1) {
          X[na_ix, col] <- mean(X[na_ix-1, col], X[na_ix+1, col])
        }
      }
    }
    which9999 <- apply(X, 1, function(x) length(which(x==-9999)))
    ix9999 <- which(which9999 > 0)
    if (length(ix9999) > 0) {
      message(sprintf('%s: rows %s contain all -9999s. Observations will be removed.',
                      fname, paste(sprintf('%s', ix9999), collapse=',')))
    }
    which0 <- apply(X, 1, function(x) length(which(x==0)))
    ix0 <- which(which0 > 0)
    if (length(ix0) > 0) {
      message(sprintf('%s: rows %s contains all zeros. Observations will be removed.',
                      fname, paste(sprintf('%s', ix0), collapse=',')))
    }
  } else {
    X <- flx_hsi_gdf[, sprintf('comp%02d', seq(1:nbands)), drop=TRUE]
  }

  # combine gap-filled bands with relevant columns from GPKG
  flx_hsi_gdf_clean <- cbind(flx_hsi_gdf[, c(response, 'name', 'date', 'PPI', 'SPEI_365', 'ecosystem', 'X', 'Y')], X)
  flx_hsi_gdf_clean <- flx_hsi_gdf_clean %>% mutate_if(is.character, as.factor)
  
  # add ecosystem type dummy variables
  #flx_hsi_gdf_clean$forest <- ifelse(grepl('MF|DBF|ENF|EBF', flx_hsi_gdf_clean$ecosystem), 1, 0)
  dummies <- model.matrix(eval(parse(text=response))~.,
                          flx_hsi_gdf_clean[, c(response,'ecosystem'), drop=TRUE])
  dummies <- dummies[, -which(colnames(dummies) == response)]
  dummies[,1] <- ifelse(flx_hsi_gdf_clean$ecosystem == levels(
    flx_hsi_gdf_clean$ecosystem)[1], 1, 0)
  dummynames <- paste0('es.', lapply(levels(as.factor(flx_hsi_gdf$ecosystem)), tolower))
  colnames(dummies) <- dummynames
  flx_hsi_gdf_f <- cbind(flx_hsi_gdf_clean, dummies)
  
  if (!is.null(site_only)) {
    flx_hsi_gdf_f = flx_hsi_gdf_f[flx_hsi_gdf_f$name == site_only, ]
  }
  
  if (!is.null(year_only)) {
    if (is.integer(year_only)) {
      year_only <- as.character(year_only)
    }
    flx_hsi_gdf_f = flx_hsi_gdf_f[grepl(year_only, flx_hsi_gdf_f$date, fixed = T), ]
    # remove all ecosystem columns with all zeros
    dropcols <- c()
    for (name in dummynames) {
      print(paste(name, 'is a dummy!'))
      if (all(flx_hsi_gdf_f[, name, drop=T] == 0)) {
        print(paste(name, 'will be removed!'))
        dropcols <- c(dropcols, name)
      }
    }
    print(dropcols)
    flx_hsi_gdf_f <- flx_hsi_gdf_f[, !(names(flx_hsi_gdf_f) %in% dropcols)]
  }
  
  if (dr == FALSE) {
    if (length(ix9999) > 0 || length(ix0) > 0) {
      flx_hsi_gdf_f <- flx_hsi_gdf_f[-union(ix9999, ix0), ]
    }
  }

  if (return_df == TRUE) {
    flx_hsi_gdf_f = flx_hsi_gdf_f[,,drop=TRUE][,-which(names(flx_hsi_gdf_f) %in% c('geom'))]
  } else {
    # TEMP: could be replaced by using coords from flx_db
    flx_hsi_gdf_f$geom <- st_centroid(flx_hsi_gdf_f) %>% 
      st_geometry()
  }
  
  return(list('hsi_df' = flx_hsi_gdf_f, 'nbands' = nbands))
}


label.feature <- function(x){
  # a saved list of some feature names that I am using
  labs <- SHAPforxgboost::labels_within_package
  # but if you supply your own `new_labels`, it will print your feature names
  # must provide a list.
  if (!is.null(new_labels)) {
    if(!is.list(new_labels)) {
      message("new_labels should be a list, for example,`list(var0 = 'VariableA')`.\n")
    }  else {
      message("Plot will use your user-defined labels.\n")
      labs = new_labels
    }
  }
  out <- rep(NA, length(x))
  for (i in 1:length(x)){
    if (is.null(labs[[ x[i] ]])){
      out[i] <- x[i]
    }else{
      out[i] <- labs[[ x[i] ]]
    }
  }
  return(out)
}

shap.plot.summary2 <- function(data_long, x_bound = NULL, dilute = FALSE,
                               scientific = FALSE, my_format = NULL){
  
  if (scientific){label_format = "%.1e"} else {label_format = "%.3f"}
  if (!is.null(my_format)) label_format <- my_format
  # check number of observations
  N_features <- data.table::setDT(data_long)[,uniqueN(variable)]
  if (is.null(dilute)) dilute = FALSE
  
  nrow_X <- nrow(data_long)/N_features # n per feature
  if (dilute!=0){
    # if nrow_X <= 10, no dilute happens
    dilute <- ceiling(min(nrow_X/10, abs(as.numeric(dilute)))) # not allowed to dilute to fewer than 10 obs/feature
    set.seed(1234)
    data_long <- data_long[sample(nrow(data_long),
                                  min(nrow(data_long)/dilute, nrow(data_long)/2))] # dilute
  }
  
  x_bound <- if (is.null(x_bound)) max(abs(data_long$value))*1.1 else as.numeric(abs(x_bound))
  plot1 <- ggplot(data = data_long) +
    coord_flip(ylim = c(-x_bound, x_bound)) +
    geom_hline(yintercept = 0) + # the y-axis beneath
    # sina plot:
    ggforce::geom_sina(aes(x = variable, y = value, color = stdfvalue),
                       method = "counts", maxwidth = 0.7, alpha = 0.7) +
    # print the mean absolute value:
    geom_text(data = unique(data_long[, c("variable", "mean_value")]),
              aes(x = variable, y=-Inf, label = sprintf(label_format, mean_value)),
              size = 5, alpha = 0.7,
              hjust = -0.2,
              fontface = "bold") + # bold
    # # add a "SHAP" bar notation
    # annotate("text", x = -Inf, y = -Inf, vjust = -0.2, hjust = 0, size = 3,
    #          label = expression(group("|", bar(SHAP), "|"))) +
    scale_color_gradient(low="#FFCC33", high="#6600CC",
                         breaks=c(0,1), labels=c(" Low","High "),
                         guide = guide_colorbar(barwidth = 12, barheight = 0.3)) +
    theme_bw() +
    theme(axis.line.y = element_blank(),
          axis.ticks.y = element_blank(), # remove axis line
          legend.position="bottom",
          legend.title=element_text(size=16),
          legend.text=element_text(size=14),
          axis.title.x= element_text(size = 16)) +
    # reverse the order of features, from high to low
    # also relabel the feature using `label.feature`
    scale_x_discrete(limits = rev(levels(data_long$variable)),
                     labels = label.feature(rev(levels(data_long$variable))))+
    labs(y = "SHAP value (impact on model output)", x = "", color = "Feature value  ")
  return(plot1)
}

