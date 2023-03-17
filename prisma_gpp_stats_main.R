## =============================================================================
##
## prisma_gpp_stats_main.R
##
## This script is used for performing benchmarking and model diagnostics on 
## combined ICOS GPP + hyperspectral data using mostly mlr3 and DALEX.
##
## Author: Floris Hermanns
##
## Date Created: 2023-03-17
##
## =============================================================================


library(tidyverse)
library(lubridate)
library(sf)
library(rhdf5)
library(RColorBrewer)
library(docstring)

library(mlr3verse)
library(visNetwork)
library(mlr3spatiotempcv)
library(paradox)
lgr::get_logger('mlr3')$set_threshold('warn')
library(iml)
library(DALEX)
library(DALEXtra)
library(SHAPforxgboost)
library(gridExtra)
# additionally: mlr3pipelines, caret, future, data.table
getCurrentFileLocation <-  function()
{
  this_file <- commandArgs() %>% 
    tibble::enframe(name = NULL) %>%
    tidyr::separate(col=value, into=c("key", "value"), sep="=", fill='right') %>%
    dplyr::filter(key == "--file") %>%
    dplyr::pull(value)
  if (length(this_file)==0)
  {
    this_file <- rstudioapi::getSourceEditorContext()$path
  }
  return(dirname(this_file))
}
wdir0 <- getCurrentFileLocation()
source(file.path(wdir0, 'R/icos_functions.R'))
plotdir <- paste0(wdir0, 'out/plots')

##### [I] Model comparison: BENCHMARK performance #####
#### [A] Create tasks: all ecosystems & forest only variants! ####
combi <- list(c('PRISMA', 'vnir', 'ffp', 'ref'),
              c('PRISMA', 'vnir', 'zst', 'ref'),
              c('PRISMA', 'vnir', 'ffp', 'ref'), # 10 PCs
              c('PRISMA', 'vnir', 'ffp', 'ref'), # 2020 only
              c('PRISMA', 'vnir', 'ffp', 'ref'), # forest only
              c('PRISMA', 'vis', 'ffp', 'ref'),
              c('PRISMA', 'vis', 'ffp', 'upw'),
              c('PRISMA', 'vswir', 'ffp', 'ref'))
for (i in 1:length(combi)) {
  names(combi[[i]]) <- c('sensor', 'sr', 'mask', 'rad') 
}
flx_gdfs <- vector(mode='list', length=length(combi))
flx_nbands <- vector(mode='integer', length=length(combi))
tasknames <- unlist(lapply(combi, paste, collapse='_'))
tasknames <- as.vector(sapply('flx', paste, tasknames, sep='_'))
# add model & ecosystem info
tasknames <- paste('v2_pca', tasknames, sep='_')
tasknames[3] <- paste(tasknames[3], '10pc', sep='_')
tasknames[4] <- paste(tasknames[4], 'yo', sep='_')
tasknames[5] <- paste(tasknames[5], 'fo', sep='_')

# additional info about response variables
for (i in 1:length(combi)) {
  combi[[i]]['resp'] <- 'GPP_DT_VUT_50'
}

for (i in 1:length(combi)) {
  if (grepl('yo', tasknames[i], fixed=T) == T) {
    args <- c(list(wdir=wdir0), combi[[i]], list(return_df=T, year_only=2020))
  } else if (grepl('fo', tasknames[i], fixed=T) == T) {
    args <- c(list(wdir=wdir0), combi[[i]], list(return_df=T, forest_only=T))
  } else {
    args <- c(list(wdir=wdir0), combi[[i]], list(return_df=T))
  }
  loaded <- do.call(load_flx_gdf, args)
  flx_gdfs[[i]] <- loaded$hsi_df
  flx_nbands[i] <- loaded$nbands
  ntask <- TaskRegr$new(tasknames[i], flx_gdfs[[i]], target = args$resp)
  #ntask <- as_task_regr_st(flx_gdfs[[i]], id=tasknames[i], target=args$resp,
  #                         positive='TRUE', coordinate_names=c('X', 'Y'), crs=3035)
  mlr_tasks$add(tasknames[i], ntask)
}
tasks = lapply(tasknames, tsk)

taskpaths <- file.path(plotdir, tasknames)
lapply(taskpaths, function(x) if(!dir.exists(x)) dir.create(x))

#### [B] Fit PCA per task to determine good ncomp for mlr3 pipeline ####
flx_pca_varexp <- vector(mode='list', length=length(combi))

for (i in 1:length(combi)) {
  bands <- c(combi[[i]]['resp'], sprintf('b%03d', seq(1:flx_nbands[i])))
  XY <- tasks[[i]]$data()[, ..bands]
  colnames(XY)[1] <- 'Y'
  myfolds <- createMultiFolds(unlist(XY[, 'Y']), k = 5)
  pca_control <- trainControl('cv', number = 5, search = 'grid', index = myfolds, 
                              selectionFunction = 'oneSE', savePredictions = 'final')
  # centering (not necessarily scaling) of hyperspectral bands before PCA leads to more robust results
  # https://opengeospatialdata.springeropen.com/articles/10.1186/s40965-017-0028-1
  pca_fit <- caret::train(Y~., data = XY, method = 'pcr', metric = 'RMSE',
                   trControl = pca_control, tuneGrid = expand.grid(ncomp = 6:10),
                   preProc = c('center', 'scale'))
  xve <- cumsum(explvar(pca_fit$finalModel))
  yve <- 100 * drop(R2(pca_fit$finalModel, estimate='train', intercept=FALSE)$val)
  flx_pca_varexp[[i]] <- data.frame(comps=names(xve), X=xve, Y=yve, row.names=NULL) %>%
    pivot_longer(!comps, names_to='typeofvar', values_to='varexp')
}

for (i in 1:length(combi)) {
  ggplot(flx_pca_varexp[[i]], aes(x=fct_inorder(comps), y=varexp)) +
    geom_bar(stat='identity', fill='forest green') +
    facet_wrap(~typeofvar) + ylab('Var. expl. [%]') + xlab(NULL) +
    theme(axis.text.x = element_text(angle = 90))
  ggsave(file.path(taskpaths[i], 'pca_screeplot.png'), device='png', scale=2,
         width=9, height=5, units='cm', dpi=150)
}

pca_ranks <- c(3,3,10,3,3,3,3,9) # determined with PCA screeplots (except #3)
# additional info about used learners for benchmarking (only added after combi elements have been used for gdf loading)
for (i in 1:length(combi)) {
  combi[[i]]['mod'] <- 'xgb'
}

#### [C] Build pipelines ####
graphs <- vector(mode='list', length=length(combi))
names(graphs) <- tasknames

for (i in 1:length(combi)) {
  pos1 <- po('select', id = 'bands', selector=selector_name(sprintf('b%03d', seq(1:flx_nbands[i]))))
  pos1 %>>%
    mlr_pipeops$get('scale', id='scale1', param_val=list(scale=TRUE, center=TRUE)) %>>%
    mlr_pipeops$get('pca', id='pca1', param_vals=list(rank.=pca_ranks[i])) -> pr1
  
  pos2 <- po('select', id='covars', selector=selector_name(c('SPEI_365', 'PPI')))
  pos2 %>>%
    mlr_pipeops$get('scale', id='scale2', param_val=list(scale=TRUE, center=TRUE)) -> pr2
  
  es_names <- colnames(flx_gdfs[[i]])[grepl('^es.', colnames(flx_gdfs[[i]]))]
  pos3 <- po('select', id='dummies', selector=selector_name(es_names))

  piper <- gunion(list(pr1, pr2, pos3)) %>>%
    mlr_pipeops$get('featureunion')
  piper$keep_results = TRUE # result of union must be kept for evaluation of tuned models
  
  graph <- piper %>>%
    mlr3pipelines::pipeline_targettrafo(mlr_pipeops$get(
      'learner', learner = lrn('regr.xgboost',
                               booster = 'gbtree',
                               eta = .01)))
  graph$param_set$values$targetmutate.trafo = function(x) sqrt(x)
  graph$param_set$values$targetmutate.inverter = function(x) list(response = x$response**2)
  graphs[[i]] <- GraphLearner$new(graph)
}
#graphs[[13]]$graph$plot(html = TRUE)

reps <- 4
outer_rsmp <- rsmp('repeated_cv', folds = 5, repeats = reps)
resamp <- rsmp('cv', folds = 5)
measure <- msr('regr.rmse')

ps_xg <- ps(
  regr.xgboost.nrounds = p_int(lower = 100, upper = 1000),
  regr.xgboost.max_depth = p_int(lower = 1, upper = 5),
  regr.xgboost.min_child_weight = p_int(lower = 1, upper = 5),
  regr.xgboost.subsample = p_dbl(lower = .4, upper = .9),
  regr.xgboost.colsample_bytree = p_dbl(lower = .5, upper = .7),
  .extra_trafo = function(x, param_set) {
    x$regr.xgboost.early_stopping_rounds = as.integer(ceiling(0.1 * x$regr.xgboost.nrounds))
    x
  }
)
ps_xg$add(ParamDbl$new('regr.xgboost.lambda', lower = .1, upper = 3))
ps_xg$add(ParamDbl$new('regr.xgboost.alpha', lower = .1, upper = 3))

ats <- vector(mode='list', length=length(combi))
for (i in 1:length(combi)) {
  ats[[i]] <- auto_tuner(
    method = 'random_search',
    learner = graphs[[i]],
    resampling = resamp,
    measure = measure,
    search_space = ps_xg,
    term_evals = 100,
    store_models = FALSE
  )
}

taskabb <- c('A1', 'A2', 'A3', 'E1', 'E2', 'B1', 'B2', 'C')

#### [D] Perform benchmark ####
design = data.table::data.table(
  task = tasks, learner = ats, resampling = list(outer_rsmp))
# Instantiate resampling manually (normally done by benchmark_grid)
set.seed(100)
design$resampling = Map(
  function(task, resampling) resampling$clone()$instantiate(task),
  task = design$task, resampling = design$resampling
)

# Parallel processing
future::plan(list('multisession', 'sequential'))
# Start benchmark
bmr <- benchmark(design, store_models = FALSE)

##### [II] Benchmark diagnostics & model evaluation #####
#### [A] Tests & plots to analyze benchmark results ####
bma <- bmr$aggregate(measure)[, c(1,3,7)]

bms <- bmr$score(measure)[, c('task_id', measure$id), with=FALSE]
nvar <- outer_rsmp$param_set$values$folds * length(tasks)
bms %>%
  mutate(task = str_sub(task_id, length(combi), -1)) %>%
  add_column(taskabb = as.factor(c(rep('A1', 5*reps), rep('A2', 5*reps), rep('A3', 5*reps), rep('E1', 5*reps),
                                   rep('E2', 5*reps), rep('B1', 5*reps), rep('B2', 5*reps), rep('C', 5*reps))),
             rad = c(rep('VNIR', 25*reps), rep('VIS', 10*reps), rep('VSWIR', 5*reps)),
             .before = measure$id) %>%
  select(-task_id) -> bms

do <- position_dodge(width = 0.6)
bmsvio <- ggplot(bms, aes(x=taskabb, y=regr.rmse, fill=rad)) +
  geom_violin(width=.75, size = 0.4, position=do, trim=T, scale='area') +
  # mean istead of median as there are only 4 values per category
  stat_summary(fun='mean', geom='point', shape=3, position=do, show.legend=F) +
  #geom_boxplot(width=0.1, position=do, outlier.colour=NA) +
  theme(legend.position='top', legend.key.size = unit(0.3, 'cm'),
        panel.background = element_blank(),
        panel.border = element_rect(fill=NA, color='grey60'),
        panel.grid.major.x = element_line(linewidth=.2, color='grey60'),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_line(linewidth=.1, color='grey80'),
        axis.ticks = element_line(linewidth=.3, color='grey60'),
        text = element_text(size=8)) +
  coord_flip() + scale_x_discrete(limits = rev(levels(bms$taskabb))) +
  geom_vline(xintercept = which(rev(levels(bms$taskabb)) == 'E1') + 0.5, col='white', lwd=2) +
  geom_vline(xintercept = which(rev(levels(bms$taskabb)) == 'E1') + 0.55, col='grey50', lwd=.3) +
  geom_vline(xintercept = which(rev(levels(bms$taskabb)) == 'E1') + 0.45, col='grey50', lwd=.3) +
  guides(fill=guide_legend(title='Spectral range:')) +
  xlab('task') + ylab('RMSE')
ggsave(file.path(plotdir, 'benchmark_violin.png'),
       bmsvio, device='png', width=10, height=7, units='cm', dpi=300)

#### [B] Plot PCA scores & loadings ####
# Fit PCAs with optimal ncomp for result plotting!
flx_pcas <- vector(mode='list', length=length(combi))
flx_pca_s <- vector(mode='list', length=length(combi))
flx_pca_l <- vector(mode='list', length=length(combi))

for (i in 1:length(combi)) {
  bands <- sprintf('b%03d', seq(1:flx_nbands[i]))
  X <- tasks[[i]]$data()[, ..bands]
  pca_fit <- prcomp(X, center = TRUE, scale. = TRUE, rank. = pca_ranks[i])
  flx_pcas[[i]] <- pca_fit
  flx_pca_s[[i]] <- pca_fit$x
  flx_pca_l[[i]] <- pca_fit$rotation
}


r4col <- unname(palette.colors(7, palette='R4'))
r4col[1] <- '#999999'

train_eco <- data.frame(flx_pca_s[[1]], ecosystem = flx_gdfs[[1]]$ecosystem)
boxp_df <- train_eco %>%
  pivot_longer(!ecosystem, names_to = 'component', values_to = 'score')

scat <- ggplot(train_eco, aes(x=PC2, y=PC3)) +
  geom_hline(yintercept=0, color='grey70') + geom_vline(xintercept=0, color='grey70') +
  geom_point(aes(fill=ecosystem), pch=21, size=2, alpha=.9) + coord_fixed() +
  ggtitle('(a)') + scale_fill_manual(values=r4col) + ylim(-5.5, 5) +
  guides(fill = guide_legend(title=element_blank(), nrow = 1, byrow = TRUE)) +
  theme_linedraw() +
  theme(legend.position=c(.5, .062), legend.spacing.y = unit(0, 'mm'),
        legend.box.background = element_rect(colour = 'black'),
        legend.margin=margin(c(0,3,1,4)),
        legend.text=element_text(size=7),
        legend.spacing.x = unit(1, 'mm'),
        axis.title=element_text(size=10),
        plot.title.position = 'plot')

boxp <- ggplot(boxp_df, aes(x=ecosystem, y=score, fill=ecosystem)) +
  geom_hline(yintercept=0, color='grey70') +
  geom_boxplot(alpha=.9, lwd=.4) + scale_fill_manual(values=r4col) +
  facet_wrap(~component) +
  labs(x = NULL, y = NULL, title = '(b)') + theme_linedraw() +
  theme(axis.text.x=element_text(size=7),
        axis.title=element_text(size=11),
        legend.position='none',
        plot.title.position = 'plot',
        strip.background=element_rect(fill='grey80'),
        strip.text=element_text(color = 'black'))

scoreplots <- grid.arrange(scat, boxp, nrow = 2, heights=c(1.4,1))
ggsave(file.path(plotdir, 'scoreplots.png'), scoreplots, device='png',
       scale=1, width=15, height=15, units='cm', dpi=300)

lolli <- ggplot(lload01, aes(x=bw, y=loading)) +
  annotate('rect', xmin = 680, xmax = 750, ymin = -Inf, ymax = Inf,
           alpha = .15,fill = 'black') +
  geom_segment(aes(x=bw, xend=bw, y=0, yend=loading), color='grey60', linewidth=.4) +
  geom_point(aes(color=comps), size=.7) + 
  facet_grid(comps~., space='free', scales='free') + theme_bw() +
  theme(legend.position='none',
        axis.text.x=element_text(size=8),
        axis.title=element_text(size=9),
        strip.background=element_rect(fill='grey80'),
        strip.text=element_text(color = 'black')) +
  xlab('wavelength [nm]') + ylab('loadings')
ggsave(file.path(plotdir, 'loadings.png'),
       lolli, device='png', width=12, height=7, units='cm', dpi=300)

#### [C] Plot feature contributions & values ####
future::plan('multisession')
for (i in 1:length(combi)) {
  set.seed(100)
  ats[[i]]$train(tasks[[i]])
}

# Calculate SHAP with DALEX
# create "fresh" model with optimized parameters
# tasks must be recreated without spatial CV info
dfs_dalex <- lapply(1:length(combi), function(x){
  as.data.table(ats[[x]]$learner$graph$pipeops$featureunion$.result$output$data())
})
dfs_dalex[[4]] <- dfs_dalex[[4]][, -c(9, 12)]

x_dalex <- vector(mode='list', length=length(combi))
y_dalex <- vector(mode='list', length=length(combi))
mods_dalex <- vector(mode='list', length=length(combi))
tasks_dalex <- vector(mode='list', length=length(combi))
for (i in 1:length(combi)) { # create data & models
  features <- c('PPI', 'SPEI365', sprintf('PC%1.0f', seq(1:pca_ranks[i])))
  # import preprocessed data from pipeline for iml/DALEX model building
  x_dalex[[i]] <- dfs_dalex[[i]][ ,.SD, .SDcols = !combi[[i]]['resp']]
  y_dalex[[i]] <- dfs_dalex[[i]][ ,.SD, .SDcols = combi[[i]]['resp']][[1]]
  mods_dalex[[i]] <- lrn('regr.xgboost',
                         booster = 'gbtree', eta = 0.01,
                         lambda = ats[[i]]$tuning_result$regr.xgboost.lambda,
                         alpha = ats[[i]]$tuning_result$regr.xgboost.alpha,
                         colsample_bytree = ats[[i]]$tuning_result$regr.xgboost.colsample_bytree,
                         early_stopping_rounds = ats[[i]]$tuning_result$regr.xgboost.early_stopping_rounds,
                         max_depth = ats[[i]]$tuning_result$regr.xgboost.max_depth,
                         min_child_weight = ats[[i]]$tuning_result$regr.xgboost.min_child_weight,
                         nrounds = ats[[i]]$tuning_result$regr.xgboost.nrounds,
                         subsample = ats[[i]]$tuning_result$regr.xgboost.subsample)
  tasks_dalex[[i]] <- TaskRegr$new(sprintf('task%02d_%s_%s', i, combi[[i]]['mod'], taskabb[i]),
                                   dfs_dalex[[i]], target = combi[[i]]['resp'])
}

exps_dalex <- vector(mode='list', length=length(combi))
shaps_long <- vector(mode='list', length=length(combi))
ps_shap <- vector(mode='list', length=length(combi))

future::plan(list('multisession', 'sequential'))
for (i in 1:length(combi)) {
  mods_dalex[[i]]$train(tasks_dalex[[i]])
  exps_dalex[[i]] <- explain_mlr3(mods_dalex[[i]], data = x_dalex[[i]], y = y_dalex[[i]],
                                  label = sprintf('exp%02d_%s_%s', i, combi[[i]]['mod'], taskabb[i]), colorize = FALSE)
  shap_values <- data.frame(matrix(0, nrow(dfs_dalex[[i]]), ncol(x_dalex[[i]])))
  
  for (j in 1:nrow(dfs_dalex[[i]])) {
    shap_single <- predict_parts(exps_dalex[[i]], type = 'shap',
                                 new_observation = as.data.frame(dfs_dalex[[i]][j, ]),)
    shap_wide <- as.data.frame(shap_single) %>%
      pivot_wider(id_cols = 'B', names_from = 'variable_name', values_from = 'contribution')
    shap_wide_aggr <- shap_wide[2:ncol(shap_wide)] %>% colMeans()
    shap_values[j, ] <- shap_wide_aggr
  }
  names(shap_values) <- names(shap_wide_aggr)
  shaps_long[[i]] <- shap.prep(shap_contrib = shap_values, X_train = x_dalex[[i]])
  ps_shap[[i]] <- shap.plot.summary(shaps_long[[i]]) + theme(text = element_text(size=24))
}

# SHAP summary plots
for (i in 1:length(combi)) {
  features <- c('PPI', 'SPEI_365', sprintf('PC%1.0f', seq(1:pca_ranks[i])))
  h <- ifelse(length(features) > 6, 32, 24)
  ggsave(file.path(taskpaths[i], sprintf('shap_summary_%s.png', combi[[i]]['mod'])),
         ps_shap[[i]], device='png', scale=.7, width=28, height=h, units='cm',
         dpi=300)
}

# ALE plots
par('mar' = c(7, 6, 6, 4))
ps_ale <- vector(mode='list', length=length(combi))
for (i in 1:length(combi)) {
  features <- c('PPI', 'SPEI_365', sprintf('PC%1.0f', seq(1:pca_ranks[i])))
  ales_dalex <- model_profile(exps_dalex[[i]], variables = features, type = 'accumulated')
  ps_ale[[i]] <- plot(ales_dalex) + theme_bw() + theme(plot.title = element_blank(),
                                                       plot.subtitle = element_blank())
}
lapply(1:length(combi), function(x){
  features <- c('PPI', 'SPEI_365', sprintf('PC%1.0f', seq(1:pca_ranks[x])))
  h <- ifelse(length(features) > 6, 28, 20)
  w <- ifelse(length(features) > 6, 44, 36)
  ggsave(file.path(taskpaths[x], sprintf('ale_%s.png', combi[[x]]['mod'])),
         ps_ale[[x]], device='png', scale=.7, width=w, height=h, units='cm', dpi=300)
})

# Residual plots
r4col <- unname(palette.colors(7, palette='R4'))
r4col[1] <- '#999999'
ix = 2 # task 2

md_dalex <- model_diagnostics(exps_dalex[[ix]])
md_dalex['eco'] <- flx_gdfs[[ix]]$ecosystem

p_sr <- ggplot(data=md_dalex, aes(x=y_hat, y=residuals)) +
  geom_point(mapping=aes(fill=eco), shape=21, size=1.5, stroke=.3, alpha=.9) +
  scale_fill_manual(values=unname(r4col), guide=guide_legend(title=NULL)) +
  geom_smooth(method='loess', se=F, color='grey60', linewidth=.5) +
  xlab(expression("predicted GPP"~group("[",µmol~CO[2]~m^{-2}~h^{-1},"]"))) +
  theme_linedraw() +
  theme(legend.text=element_text(size=8),
        axis.text=element_text(size=8),
        axis.title=element_text(size=9),
        legend.spacing.x = unit(.1, 'mm'))
ggsave(file.path(taskpaths[ix], sprintf('resids_%s.png', combi[[ix]]['mod'])),
       p_sr, device='png', scale=1, width=12, height=7, units='cm', dpi=300)


plot(md_dalex, variable = 'PPI') + theme_bw() + theme(plot.title = element_blank(),
                                                      plot.subtitle = element_blank())

# Residual boxplot
resids <- lapply(exps_dalex, model_performance)
for (i in 1:length(combi)) {
  if (i %in% c(4,5)) {
    resids[[i]]$residuals$group <- rep('B', nrow(flx_gdfs[[i]]))
  } else {
    resids[[i]]$residuals$group <- rep('A', nrow(flx_gdfs[[i]]))
  }
}

p_resid <- plot(resids, geom='boxplot') +
  scale_x_discrete(limits = c('exp05_xgb_E2', 'exp04_xgb_E1', 'exp08_xgb_C', 'exp07_xgb_B2',
                              'exp06_xgb_B1', 'exp03_xgb_A3', 'exp02_xgb_A2', 'exp01_xgb_A1'),
                   labels = c('E2', 'E1', 'C', 'B2', 'B1', 'A3', 'A2', 'A1')) +
  geom_vline(xintercept = which(levels(bms$taskabb) == 'A3') - 0.5, col='grey30', lwd=.4) +
  guides(fill='none') + ylab('absolute residuals') + xlab('task') +
  theme_bw() + theme(plot.title = element_blank(),
                     plot.subtitle = element_blank(),
                     panel.grid.major.y = element_blank())
p_resid$layers[[2]]$aes_params$size <- 2
p_resid
ggsave(file.path(plotdir, 'resid_boxplot.png'),
      p_resid, device='png', scale=1, width=12, height=6, units='cm', dpi=300)
